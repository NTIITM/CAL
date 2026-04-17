"""
5-Fold Stratified Cross-Validation for CellFM + CAL
=====================================================
Usage:
    python kfold_cellfm.py --dataset MS --mode baseline --cuda 0 --epochs 10
    python train_cellfm.py --dataset MS --mode cal      --cuda 0 --epochs 10

Output:
    results/result_cellfm_{dataset}/{mode}/kfold_results.json   <- per-fold scores
    results/result_cellfm_{dataset}/{mode}/kfold_summary.json   <- mean ± std
"""

import os, sys, warnings, json, pathlib, random, logging, time
warnings.filterwarnings("ignore")

# Add paths
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "pretrained_models" / "CellFM_torch"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
import scanpy as sc

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

from model import Finetune_Cell_FM
from layers.utils import Config_80M, SCrna, Prepare, build_dataset
from cal_loss import CALLoss, MultiHeadCALLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Args ────────────────────────────────────────────────────────────────────
def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',      type=str,   default='MS')
    p.add_argument('--mode',         type=str,   default='baseline', choices=['baseline', 'cal', 'mhcal', 'mhcal_lora'])
    p.add_argument('--cuda',         type=int,   default=0)
    p.add_argument('--lambda_attn',  type=float, default=0.5, help="CAL loss weight")
    p.add_argument('--epochs',       type=int,   default=10)
    p.add_argument('--batch_size',   type=int,   default=16)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=5)
    p.add_argument('--n_folds',      type=int,   default=5)
    p.add_argument('--accum_steps',  type=int,   default=1, help="Gradient accumulation steps")
    return p.parse_args()


# ─── Train / Eval ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, criterion_cls, cal_loss_fn, device, alpha, accum_steps=1, mode='baseline'):
    model.train()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        raw_nzdata = batch['raw_nzdata'].to(device)
        dw_nzdata  = batch['dw_nzdata'].to(device)
        ST_feat    = batch['ST_feat'].to(device)
        nonz_gene  = batch['nonz_gene'].to(device)
        mask_gene  = batch['mask_gene'].to(device)
        zero_idx   = batch['zero_idx'].to(device)
        feat       = batch['feat'].long().to(device)

        with autocast():
            if alpha > 0:
                cls, mask_loss, cls_token, weights = model(
                    raw_nzdata=raw_nzdata, dw_nzdata=dw_nzdata, ST_feat=ST_feat,
                    nonz_gene=nonz_gene, mask_gene=mask_gene, zero_idx=zero_idx,
                    return_weights=True
                )
                cls_loss = criterion_cls(cls, feat)
                # Retention weights are unbounded inner products — normalize to
                # probability distributions (like Transformer softmax attention)
                # so CAL's contrastive learning operates in the correct space
                cls_weights = weights[:, :, 0, 1:]          # (B, H, L-1)
                cls_weights = F.softmax(cls_weights, dim=-1) # normalize per-head
                
                if mode in ('mhcal', 'mhcal_lora'):
                    # Multi-Head independent contrastive loss
                    cal_loss = cal_loss_fn(cls_weights, feat, None)
                else:
                    # Original CAL with head-averaging
                    attn_map = cls_weights.mean(dim=1)
                    cal_loss = cal_loss_fn(attn_map, feat, None)
            else:
                cls, mask_loss, _ = model(
                    raw_nzdata=raw_nzdata, dw_nzdata=dw_nzdata, ST_feat=ST_feat,
                    nonz_gene=nonz_gene, mask_gene=mask_gene, zero_idx=zero_idx,
                    return_weights=False
                )
                cls_loss = criterion_cls(cls, feat)
                cal_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + (mask_loss if not torch.isnan(mask_loss) else 0.0) + alpha * cal_loss
            loss = loss / accum_steps  # Scale loss for accumulation

        # NaN guard
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("  NaN/Inf loss detected, skipping batch")
            continue

        scaler.scale(loss).backward()

        # Step optimizer every accum_steps batches
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        acc = (cls.argmax(1) == feat).sum().item() / feat.size(0)
        running_loss += loss.item() * accum_steps  # Undo scaling for logging
        running_acc += acc
        n_batches += 1

    if n_batches == 0:
        return float('nan'), 0.0
    return running_loss / n_batches, running_acc / n_batches


@torch.no_grad()
def eval_epoch(model, loader, criterion_cls, device):
    model.eval()
    running_loss, n_batches = 0.0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        raw_nzdata = batch['raw_nzdata'].to(device)
        dw_nzdata  = batch['dw_nzdata'].to(device)
        ST_feat    = batch['ST_feat'].to(device)
        nonz_gene  = batch['nonz_gene'].to(device)
        mask_gene  = batch['mask_gene'].to(device)
        zero_idx   = batch['zero_idx'].to(device)
        feat       = batch['feat'].long().to(device)

        with autocast():
            result = model(
                raw_nzdata=raw_nzdata, dw_nzdata=dw_nzdata, ST_feat=ST_feat,
                nonz_gene=nonz_gene, mask_gene=mask_gene, zero_idx=zero_idx
            )
            cls = result[0]
            # mask_loss may be a tuple in eval mode; we only need cls_loss for metrics
            loss = criterion_cls(cls, feat)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            running_loss += loss.item()
            n_batches += 1
        all_preds.extend(cls.argmax(1).cpu().numpy())
        all_labels.extend(feat.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc_metric = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / max(n_batches, 1)
    return avg_loss, f1, acc_metric


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} | Mode: {args.mode}")

    dataset_path = str(PROJECT_ROOT / "data" / args.dataset / f"{args.dataset}.h5ad")
    ckpt_path = str(PROJECT_ROOT / "pretrained_models" / "CellFM_weights" / "CellFM_80M_weight.pth")

    logger.info(f"Loading dataset: {dataset_path}")
    adata = sc.read_h5ad(dataset_path)

    # ── Convert Ensembl IDs to HGNC gene symbols if needed ──
    if adata.var_names[0].startswith("ENSG"):
        if 'gene_name' in adata.var.columns:
            logger.info("  Detected Ensembl IDs → converting to HGNC symbols via var['gene_name']")
            new_names = adata.var['gene_name'].astype(str).values
            # Deduplicate gene names (append suffix for duplicates)
            from collections import Counter

            name_counts = Counter(new_names)
            seen = {}
            deduped = []
            for n in new_names:
                if name_counts[n] > 1:
                    seen[n] = seen.get(n, 0) + 1
                    deduped.append(f"{n}_{seen[n]}")
                else:
                    deduped.append(n)
            adata.var_names = deduped
            adata.var_names_make_unique()
            logger.info(f"  After conversion: var_names sample = {list(adata.var_names[:5])}")
        else:
            logger.warning("  var_names are Ensembl IDs but no 'gene_name' column found — gene mapping will fail!")

    # Map classes
    label_col = 'cell_type' if 'cell_type' in adata.obs.columns else 'celltype'
    if label_col not in adata.obs.columns:
        label_col = 'Celltype' if 'Celltype' in adata.obs.columns else 'Celltype2'

    le = LabelEncoder()
    adata.obs['celltype'] = adata.obs[label_col].astype(str)
    adata.obs['feat'] = le.fit_transform(adata.obs['celltype'])
    y = adata.obs['feat'].values
    num_cls = len(np.unique(y))
    logger.info(f"  {len(y)} cells | {num_cls} classes")

    out_dir = pathlib.Path(str(PROJECT_ROOT / "results" / f"result_cellfm_{args.dataset}" / args.mode))
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    indices = np.arange(len(adata))

    fold_results = []
    total_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(indices, y)):
        fold_start = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold_idx+1}/{args.n_folds}  mode={args.mode}")

        # Seeds
        random.seed(42 + fold_idx)
        np.random.seed(42 + fold_idx)
        torch.manual_seed(42 + fold_idx)
        torch.cuda.manual_seed_all(42 + fold_idx)

        train_adata = adata[train_idx].copy()
        test_adata = adata[test_idx].copy()

        train_adata.obs['train'] = 0
        test_adata.obs['train'] = 2

        cfg = Config_80M()
        cfg.num_cls = num_cls
        cfg.ckpt_path = ckpt_path
        cfg.device = device
        cfg.use_bs = args.batch_size
        cfg.ecs_threshold = 0.8
        cfg.ecs = True
        cfg.add_zero = True
        cfg.pad_zero = True

        # Enable LoRA in retention layers for mhcal_lora mode
        if args.mode == 'mhcal_lora':
            cfg.lora = 8  # LoRA rank

        train_ds = SCrna(train_adata, mode="train")
        test_ds = SCrna(test_adata, mode="test")
        prep = Prepare(cfg.nonz_len, pad=0, mask_ratio=0.5)

        train_loader = build_dataset(train_ds, prep, batch_size=cfg.use_bs, shuffle=True)
        test_loader = build_dataset(test_ds, prep, batch_size=cfg.use_bs, shuffle=False)

        model = Finetune_Cell_FM(cfg)
        model.extractor.load_model(weight=True, moment=False)
        model.to(device)

        if args.mode == 'mhcal_lora':
            # Freeze ALL base encoder parameters, only train LoRA + classifier
            for name, param in model.extractor.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
            # Count trainable params
            lora_params = [p for n, p in model.extractor.named_parameters() if 'lora' in n and p.requires_grad]
            cls_params = list(model.cls.parameters())
            n_lora = sum(p.numel() for p in lora_params)
            n_cls = sum(p.numel() for p in cls_params)
            logger.info(f"  LoRA params: {n_lora:,} | Classifier params: {n_cls:,} | Total trainable: {n_lora+n_cls:,}")
            optimizer = AdamW([
                {'params': lora_params, 'lr': args.lr},
                {'params': cls_params, 'lr': args.lr}
            ], weight_decay=1e-5)
        else:
            head_params = list(model.cls.parameters())
            encoder_params = list(model.extractor.parameters())
            optimizer = AdamW([
                {'params': encoder_params, 'lr': args.lr * 0.1},
                {'params': head_params, 'lr': args.lr}
            ], weight_decay=1e-5)

        # Safer AMP scaler
        scaler = GradScaler(init_scale=2**10, growth_interval=2000)
        criterion_cls = nn.CrossEntropyLoss()
        if args.mode in ('mhcal', 'mhcal_lora'):
            cal_loss_fn = MultiHeadCALLoss(lambda_attn=1.0, temperature=0.1).to(device)
            alpha = args.lambda_attn
        else:
            cal_loss_fn = CALLoss(lambda_attn=1.0, temperature=0.1, use_class_balanced_queue=False).to(device)
            alpha = args.lambda_attn if args.mode == 'cal' else 0.0

        best_val_f1 = 0.0
        patience_cnt = 0
        best_model_path = out_dir / f"best_model_fold{fold_idx}.pth"

        for epoch in range(1, args.epochs + 1):
            ep_start = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, criterion_cls, cal_loss_fn, device, alpha, args.accum_steps, mode=args.mode)
            val_loss, val_f1, val_acc = eval_epoch(model, test_loader, criterion_cls, device)
            ep_time = time.time() - ep_start

            logger.info(f"  Fold {fold_idx+1} Ep {epoch:3d} | val_f1={val_f1:.4f} val_acc={val_acc:.4f} TrLoss={train_loss:.4f} ({ep_time:.1f}s)")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_cnt = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    logger.info(f"  Early stop at epoch {epoch}")
                    break

        # Final evaluation
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
        _, test_f1, test_acc = eval_epoch(model, test_loader, criterion_cls, device)
        fold_time = time.time() - fold_start
        logger.info(f"  >>> FOLD {fold_idx+1} TEST: F1={test_f1:.4f} Acc={test_acc:.4f} ({fold_time:.0f}s)")

        fold_results.append({
            "fold": fold_idx + 1,
            "test_f1": float(test_f1),
            "test_acc": float(test_acc),
            "dataset": args.dataset,
            "mode": args.mode,
            "fold_time_sec": round(fold_time, 1),
        })

        with open(out_dir / "kfold_results.json", "w") as f:
            json.dump(fold_results, f, indent=2)

    # Summary
    total_time = time.time() - total_start
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
        "total_time_sec": round(total_time, 1),
    }
    with open(out_dir / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"[{args.dataset}] [{args.mode}] 5-Fold CV Summary:")
    print(f"  Macro F1  = {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Accuracy  = {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Total time: {total_time/60:.1f} min")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
