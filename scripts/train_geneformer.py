"""
5-Fold Stratified Cross-Validation for Geneformer + CAL
=========================================================
Usage:
    python kfold_geneformer.py --dataset human_CD4 --mode baseline --cuda 2 --epochs 50
    python train_geneformer.py --dataset hBone --mode cal     --cuda 0 --epochs 50

Controlled variables (identical to single-run CellClassfication.py):
    - LoRA: r=8, alpha=16, target_modules=["query","value"], dropout=0.1
    - Optimizer: AdamW, lr=1e-4
    - CAL: lambda_attn=0.5, temperature=0.1
    - patience=5 (early stopping on val_loss)
    - max_len=512 (truncated from 1024 for efficiency)
    - batch_size=8 (increased from 2, enabled by shorter sequences)
"""

import sys, warnings, json, pathlib, random, logging
warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "pretrained_models" / "Geneformer"))
sys.path.insert(0, str(BASE_DIR / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig
from collections import Counter

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent


# ─── Args ────────────────────────────────────────────────────────────────────
def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',      type=str,   default='hBone')
    p.add_argument('--mode',         type=str,   default='baseline')
    p.add_argument('--cuda',         type=int,   default=0)
    p.add_argument('--lambda_attn',  type=float, default=0.5)
    p.add_argument('--lambda_orth',  type=float, default=0.1)
    p.add_argument('--temperature',  type=float, default=0.1)
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=5)
    p.add_argument('--max_len',      type=int,   default=512)
    p.add_argument('--n_folds',      type=int,   default=5)
    return p.parse_args()


# ─── Dataset ─────────────────────────────────────────────────────────────────
class GeneformerDataset(Dataset):
    """Wraps a list of (input_ids, label_id) pairs."""
    def __init__(self, input_ids_list, labels_list, max_len=512):
        self.input_ids_list = input_ids_list
        self.labels_list = labels_list
        self.max_len = max_len

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        ids = self.input_ids_list[idx][:self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels_list[idx], dtype=torch.long)


# ─── Model builder ───────────────────────────────────────────────────────────
def build_model(num_classes, device):
    model_dir = str(BASE_DIR / "pretrained_models/Geneformer")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=num_classes)

    # LoRA config — identical to original pipeline
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"]
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    return model


# ─── Train / Eval ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device, cal_loss_fn=None, use_cal=False):
    model.train()
    total_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(input_ids=inputs, labels=labels, output_attentions=use_cal)
        loss = outputs.loss

        if use_cal and cal_loss_fn is not None and outputs.attentions is not None:
            raw_attn = outputs.attentions[-1]  # (B, H, S, S) — keep all heads!
            pad_mask = (inputs == 0)
            loss = loss + cal_loss_fn(raw_attn, labels, src_key_padding_mask=pad_mask)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(input_ids=inputs, labels=labels)
        total_loss += outputs.loss.item()
        all_preds.extend(outputs.logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), f1, acc


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading dataset: {args.dataset}")
    ds_dir = BASE_DIR / "data" / args.dataset / "geneformer_ds"
    train_ds = load_from_disk(str(ds_dir / f"{args.dataset}_train.dataset"))
    test_ds  = load_from_disk(str(ds_dir / f"{args.dataset}_test.dataset"))

    # Merge train + test for proper K-fold
    full_ds = concatenate_datasets([train_ds, test_ds])

    # Extract arrays
    all_input_ids = [item['input_ids'] for item in full_ds]
    all_cell_types = [item['cell_type'] for item in full_ds]

    # Build label mapping
    unique_labels = sorted(set(all_cell_types))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    all_labels = np.array([label_map[ct] for ct in all_cell_types])

    # Filter singleton classes
    min_samples = args.n_folds * 2
    class_counts = Counter(all_labels)
    valid_mask = np.array([class_counts[l] >= min_samples for l in all_labels])
    dropped = sum(1 for c in class_counts.values() if c < min_samples)
    if dropped > 0:
        logger.warning(f"  Dropping {dropped} classes with <{min_samples} samples")
        all_input_ids = [all_input_ids[i] for i in range(len(all_input_ids)) if valid_mask[i]]
        all_labels = all_labels[valid_mask]
        # Re-encode
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        all_labels = le.fit_transform(all_labels)

    num_classes = len(np.unique(all_labels))
    logger.info(f"  {len(all_labels)} cells | {num_classes} classes | max_len={args.max_len} | bs={args.batch_size}")

    # Output dir
    out_dir = BASE_DIR / "results" / f"result_gf_{args.dataset}" / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    from cal_loss import MultiHeadCALLoss

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(all_input_ids, all_labels)):
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold_idx+1}/{args.n_folds}  mode={args.mode}")

        random.seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)
        torch.manual_seed(42 + fold_idx)
        torch.cuda.manual_seed_all(42 + fold_idx)

        # Split train -> sub_train + val (15%)
        sub_train_idx, val_idx = train_test_split(
            train_idx, test_size=0.15,
            random_state=42 + fold_idx,
            stratify=all_labels[train_idx])

        def make_loader(indices, shuffle):
            ids  = [all_input_ids[i] for i in indices]
            labs = all_labels[indices].tolist()
            ds = GeneformerDataset(ids, labs, max_len=args.max_len)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

        train_loader = make_loader(sub_train_idx, shuffle=True)
        val_loader   = make_loader(val_idx,       shuffle=False)
        test_loader  = make_loader(test_idx,      shuffle=False)

        # Fresh model per fold
        model = build_model(num_classes, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        cal_loss_fn = None
        use_cal = (args.mode in ('cal', 'mhcal', 'mhcal_orth'))
        if args.mode in ('cal', 'mhcal'):
            cal_loss_fn = MultiHeadCALLoss(lambda_attn=args.lambda_attn, temperature=args.temperature).to(device)
        elif args.mode == 'mhcal_orth':
            cal_loss_fn = MultiHeadCALLoss(lambda_attn=args.lambda_attn, lambda_orth=args.lambda_orth, temperature=args.temperature).to(device)

        best_val_loss = float('inf')
        patience_cnt = 0
        best_path = out_dir / f"best_model_fold{fold_idx}.pt"

        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, device,
                        cal_loss_fn=cal_loss_fn, use_cal=use_cal)
            val_loss, val_f1, val_acc = eval_epoch(model, val_loader, device)
            logger.info(f"  Fold {fold_idx+1} Ep {epoch:3d} | val_loss={val_loss:.4f} val_f1={val_f1:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

        # Test fold
        model.load_state_dict(torch.load(best_path))
        _, test_f1, test_acc = eval_epoch(model, test_loader, device)
        logger.info(f"  >>> FOLD {fold_idx+1} TEST: F1={test_f1:.4f} Acc={test_acc:.4f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "test_f1": float(test_f1),
            "test_acc": float(test_acc),
            "dataset": args.dataset,
            "mode": args.mode,
        })
        with open(out_dir / "kfold_results.json", "w") as f:
            json.dump(fold_results, f, indent=2)

        # Free GPU memory between folds
        del model, optimizer
        if cal_loss_fn: del cal_loss_fn
        torch.cuda.empty_cache()

    # Summary
    f1s  = [r["test_f1"]  for r in fold_results]
    accs = [r["test_acc"] for r in fold_results]
    summary = {
        "dataset": args.dataset, "mode": args.mode,
        "n_folds": args.n_folds, "max_len": args.max_len,
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
        "f1_per_fold": f1s, "acc_per_fold": accs,
    }
    with open(out_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[{args.dataset}] [{args.mode}] Geneformer 5-Fold CV Summary:")
    print(f"  Macro F1  = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Accuracy  = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
