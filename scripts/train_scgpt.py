"""
5-Fold Stratified Cross-Validation for scGPT + CAL
=====================================================
Usage:
    python kfold_scgpt.py --dataset hBone --mode baseline --cuda 3 --epochs 80
    python train_scgpt.py --dataset MS    --mode cal     --cuda 0 --epochs 80

Output:
    results/result_scgpt_{dataset}/{mode}/kfold_results.json   <- per-fold scores
    results/result_scgpt_{dataset}/{mode}/kfold_summary.json   <- mean ± std
"""
import sys, warnings, json, pathlib, random, logging
warnings.filterwarnings("ignore")
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pretrained_models" / "scGPT_repo"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cal_loss import MultiHeadCALLoss

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from scipy.sparse import issparse

from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch
import scanpy as sc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─── LoRA Adapter ────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """Low-Rank Adapter for a weight matrix."""
    def __init__(self, in_features: int, out_features: int, rank: int = 8):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        return x @ self.lora_A.T @ self.lora_B.T  # (..., out_features)


def inject_lora(model, rank=8):
    """
    Inject LoRA adapters into every TransformerEncoderLayer's self_attn.
    We wrap the MHA forward to add LoRA delta to Q/K/V projections.
    """
    d_model = model.d_model
    lora_modules = []
    
    for layer_idx, layer in enumerate(model.transformer_encoder.layers):
        mha = layer.self_attn
        # Create LoRA adapters for Q, K, V (each d_model -> d_model)
        lora_q = LoRALinear(d_model, d_model, rank).to(next(mha.parameters()).device)
        lora_k = LoRALinear(d_model, d_model, rank).to(next(mha.parameters()).device)
        lora_v = LoRALinear(d_model, d_model, rank).to(next(mha.parameters()).device)
        
        # Store as named modules so they appear in model.parameters()
        setattr(layer, f'lora_q', lora_q)
        setattr(layer, f'lora_k', lora_k)
        setattr(layer, f'lora_v', lora_v)
        lora_modules.extend([lora_q, lora_k, lora_v])
        
        # Wrap the MHA forward to inject LoRA deltas
        orig_forward = mha.forward
        
        def make_patched_forward(orig_fn, lq, lk, lv):
            def patched_forward(query, key, value, **kwargs):
                query = query + lq(query)
                key   = key + lk(key)
                value = value + lv(value)
                return orig_fn(query, key, value, **kwargs)
            return patched_forward
        
        mha.forward = make_patched_forward(orig_forward, lora_q, lora_k, lora_v)
    
    logger.info(f"  Injected LoRA (rank={rank}) into {len(model.transformer_encoder.layers)} layers")
    return lora_modules


def freeze_pretrained(model):
    """
    Freeze all parameters except LoRA adapters and the classification head.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA adapters
    for layer in model.transformer_encoder.layers:
        for name in ['lora_q', 'lora_k', 'lora_v']:
            if hasattr(layer, name):
                for param in getattr(layer, name).parameters():
                    param.requires_grad = True
    
    # Unfreeze classification decoder
    for param in model.cls_decoder.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Frozen: {total - trainable}/{total} params | Trainable: {trainable} ({100*trainable/total:.1f}%)")


# ─── Args ────────────────────────────────────────────────────────────────────
def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',      type=str,   default='hBone')
    p.add_argument('--mode',         type=str,   default='baseline')
    p.add_argument('--cuda',         type=int,   default=0)
    p.add_argument('--lambda_attn',  type=float, default=0.5)
    p.add_argument('--lambda_orth',  type=float, default=0.1)
    p.add_argument('--epochs',       type=int,   default=80)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--patience',     type=int,   default=8)
    p.add_argument('--n_folds',      type=int,   default=5)
    return p.parse_args()


# ─── Data loading ────────────────────────────────────────────────────────────
def load_adata(dataset):
    """Load single-cell expression data for scGPT."""
    path = PROJECT_ROOT / "data" / dataset / f"{dataset}.h5ad"
    adata = sc.read_h5ad(str(path))
    return adata


# ─── Model loader ────────────────────────────────────────────────────────────
def load_model(vocab, margs, num_classes, device):
    import json as _json
    model_dir = str(PROJECT_ROOT / "pretrained_models" / "scGPT_repo" / "scgpt" / "checkpoint")
    model_file = f"{model_dir}/best_model.pt"

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=margs["embsize"],
        nhead=margs["nheads"],
        d_hid=margs["d_hid"],
        nlayers=margs["nlayers"],
        nlayers_cls=3,
        n_cls=num_classes,
        vocab=vocab,
        dropout=margs.get("dropout", 0.2),
        pad_token="<pad>",
        pad_value=margs.get("pad_value", -2),
        do_mvc=margs.get("do_mvc", False),
        do_dab=margs.get("do_dab", False),
        use_batch_labels=margs.get("use_batch_labels", False),
        num_batch_labels=margs.get("num_batch_labels", None),
        domain_spec_batchnorm=margs.get("domain_spec_batchnorm", False),
        input_emb_style=margs.get("input_emb_style", "continuous"),
        n_input_bins=margs.get("n_input_bins", None),
        cell_emb_style=margs.get("cell_emb_style", "cls"),
        mvc_decoder_style=margs.get("mvc_decoder_style", "inner product"),
        ecs_threshold=margs.get("ecs_threshold", 0.3),
        explicit_zero_prob=margs.get("explicit_zero_prob", False),
        use_fast_transformer=False,
        fast_transformer_backend="flash",
        pre_norm=margs.get("pre_norm", False),
    )
    pretrained_dict = torch.load(model_file, map_location='cpu')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    return model


# ─── Attention monkeypatch ───────────────────────────────────────────────────
def register_attn_hook(model):
    last_layer = model.transformer_encoder.layers[-1]
    orig_forward = last_layer.self_attn.forward
    captured = {}

    def patched_forward(query, key, value, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False  # Keep per-head weights!
        attn_out, attn_weights = orig_forward(query, key, value, **kwargs)
        captured['attn'] = attn_weights  # (B, H, L, L)
        return attn_out, attn_weights

    last_layer.self_attn.forward = patched_forward
    return captured


# ─── Tokenize ────────────────────────────────────────────────────────────────
def tokenize_split(X_mat, gene_ids, vocab, margs):
    tok = tokenize_and_pad_batch(
        data=X_mat,
        gene_ids=gene_ids,
        max_len=margs.get("max_seq_len", 1200),
        vocab=vocab,
        pad_token="<pad>",
        pad_value=margs.get("pad_value", -2),
        append_cls=True,
        include_zero_gene=False,
    )
    return (tok['genes'].clone().detach().long(),
            tok['values'].clone().detach().float())


# ─── Train / Eval ────────────────────────────────────────────────────────────
def train_epoch(model, vocab, loader, optimizer, criterion, device, cal_loss_fn=None, captured_attn=None):
    model.train()
    for batch_idx, (gene_ids, gene_expr, labels) in enumerate(loader):
        gene_ids, gene_expr, labels = gene_ids.to(device), gene_expr.to(device), labels.to(device)
        optimizer.zero_grad()
        src_key_padding_mask = gene_ids.eq(vocab['<pad>'])
        outputs_dict = model(gene_ids, gene_expr, src_key_padding_mask=src_key_padding_mask, CLS=True)
        outputs = outputs_dict["cls_output"]
        ce_loss = criterion(outputs, labels)
        loss = ce_loss
        if cal_loss_fn is not None and captured_attn and 'attn' in captured_attn:
            raw_attn = captured_attn['attn']  # (B, H, L, L) — per-head
            cal_loss = cal_loss_fn(raw_attn, labels, src_key_padding_mask=src_key_padding_mask)
            loss = ce_loss + cal_loss
            if batch_idx == 0:
                logger.info(f"    [loss] CE={ce_loss.item():.4f}  MH-CAL={cal_loss.item():.4f}  total={loss.item():.4f}")
        loss.backward()
        optimizer.step()


@torch.no_grad()
def eval_epoch(model, vocab, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for gene_ids, gene_expr, labels in loader:
        gene_ids, gene_expr, labels = gene_ids.to(device), gene_expr.to(device), labels.to(device)
        src_key_padding_mask = gene_ids.eq(vocab['<pad>'])
        outputs = model(gene_ids, gene_expr, src_key_padding_mask=src_key_padding_mask, CLS=True)["cls_output"]
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), f1, acc


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    import json as _json
    model_dir = str(PROJECT_ROOT / "pretrained_models" / "scGPT_repo" / "scgpt" / "checkpoint")
    with open(f"{model_dir}/args.json") as f:
        margs = _json.load(f)
    from scgpt.tokenizer import GeneVocab
    vocab = GeneVocab.from_file(f"{model_dir}/vocab.json")

    # Load entire dataset
    logger.info(f"Loading dataset: {args.dataset}")
    adata = load_adata(args.dataset)

    # Find cell type column
    for col in ["Celltype2", "Celltype", "celltype", "label", "cell_type"]:
        if col in adata.obs.columns:
            cell_type_col = col; break

    # Filter to genes in vocab
    valid_genes = [g for g in adata.var_names if g in vocab]
    adata = adata[:, valid_genes].copy()
    gene_ids_arr = np.array([vocab[g] for g in adata.var_names], dtype=int)

    le = LabelEncoder()
    all_labels_encoded = le.fit_transform(adata.obs[cell_type_col].astype(str))

    X = adata.X.toarray() if issparse(adata.X) else np.array(adata.X)
    y = all_labels_encoded

    # Filter out classes with too few samples for stratified splitting
    min_samples = args.n_folds * 2  # need at least 2*k for k-fold + internal val split
    from collections import Counter
    class_counts = Counter(y)
    valid_mask = np.array([class_counts[label] >= min_samples for label in y])
    dropped = sum(1 for c in class_counts.values() if c < min_samples)
    if dropped > 0:
        logger.warning(f"  Dropping {dropped} classes with <{min_samples} samples")
        X = X[valid_mask]
        y = y[valid_mask]
        # Re-encode to ensure contiguous labels
        le2 = LabelEncoder()
        y = le2.fit_transform(y)

    num_classes = len(np.unique(y))
    logger.info(f"  {len(y)} cells | {num_classes} classes | {X.shape[1]} genes")

    # Output dir
    out_dir = pathlib.Path(str(PROJECT_ROOT / "results" / f"result_scgpt_{args.dataset}" / args.mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold_idx+1}/{args.n_folds}  mode={args.mode}")

        # Set seeds per fold
        random.seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)
        torch.manual_seed(42 + fold_idx)
        torch.cuda.manual_seed_all(42 + fold_idx)

        # Further split train -> train + val (use 15% of train as val)
        from sklearn.model_selection import train_test_split as tts

        sub_train_idx, val_idx = tts(train_idx, test_size=0.15,
                                     random_state=42+fold_idx,
                                     stratify=y[train_idx])

        def make_loader(indices, shuffle):
            gids, gvals = tokenize_split(X[indices], gene_ids_arr, vocab, margs)
            labels_t = torch.tensor(y[indices], dtype=torch.long)
            ds = TensorDataset(gids, gvals, labels_t)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

        train_loader = make_loader(sub_train_idx, shuffle=True)
        val_loader   = make_loader(val_idx,       shuffle=False)
        test_loader  = make_loader(test_idx,      shuffle=False)

        # Fresh model per fold
        model = load_model(vocab, margs, num_classes, device)
        criterion = nn.CrossEntropyLoss()

        if args.mode == 'lora_mhcal':
            # Inject LoRA adapters and freeze pretrained weights
            inject_lora(model, rank=8)
            freeze_pretrained(model)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
        elif args.mode == 'mhcal_adamw':
            # Full fine-tuning with AdamW (same lr as SGD baseline)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        elif args.mode == 'mhcal_attn':
            # Freeze everything except attention params + classifier head
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze self_attn in every TransformerEncoderLayer
            for layer in model.transformer_encoder.layers:
                for param in layer.self_attn.parameters():
                    param.requires_grad = True
            # Unfreeze classifier head
            for param in model.cls_decoder.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(f"  [mhcal_attn] Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(trainable_params, lr=1e-3)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        cal_loss_fn = None
        captured_attn = None
        if args.mode in ('cal', 'mhcal', 'lora_mhcal', 'mhcal_adamw', 'mhcal_attn'):
            cal_loss_fn = MultiHeadCALLoss(lambda_attn=args.lambda_attn, temperature=0.1).to(device)
            captured_attn = register_attn_hook(model)
        elif args.mode == 'mhcal_orth':
            cal_loss_fn = MultiHeadCALLoss(lambda_attn=args.lambda_attn, lambda_orth=args.lambda_orth, temperature=0.1).to(device)
            captured_attn = register_attn_hook(model)

        best_val_loss = float('inf')
        patience_cnt = 0
        best_model_path = out_dir / f"best_model_fold{fold_idx}.pth"

        for epoch in range(1, args.epochs + 1):
            train_epoch(model, vocab, train_loader, optimizer, criterion, device,
                        cal_loss_fn=cal_loss_fn, captured_attn=captured_attn)
            val_loss, val_f1, val_acc = eval_epoch(model, vocab, val_loader, criterion, device)
            logger.info(f"  Fold {fold_idx+1} Ep {epoch:3d} | val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

        # Evaluate on test fold
        model.load_state_dict(torch.load(best_model_path))
        _, test_f1, test_acc = eval_epoch(model, vocab, test_loader, criterion, device)
        logger.info(f"  >>> FOLD {fold_idx+1} TEST: F1={test_f1:.4f} Acc={test_acc:.4f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "test_f1": float(test_f1),
            "test_acc": float(test_acc),
            "dataset": args.dataset,
            "mode": args.mode,
        })

        # Save incremental results
        with open(out_dir / "kfold_results.json", "w") as f:
            json.dump(fold_results, f, indent=2)

    # Summary
    f1s  = [r["test_f1"]  for r in fold_results]
    accs = [r["test_acc"] for r in fold_results]
    summary = {
        "dataset": args.dataset,
        "mode": args.mode,
        "n_folds": args.n_folds,
        "f1_mean":  float(np.mean(f1s)),
        "f1_std":   float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_per_fold": f1s,
        "acc_per_fold": accs,
    }
    with open(out_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[{args.dataset}] [{args.mode}] 5-Fold CV Summary:")
    print(f"  Macro F1  = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Accuracy  = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
