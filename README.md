# Contrastive Attention Learning (CAL) Framework

> **Official implementation for reproducing the core model training results from:**  
> *State-adaptive contrastive attention learning drives mechanistic decoupling in single-cell foundation models (MH-CAL)*

---

## Overview

Single-cell Foundation Models (scFMs) suffer from *attention collapse* — their attention heads converge to homogeneous distributions, losing cell-type discriminative capacity. The CAL framework injects **explicit structural guidance** (Eqs. 8–10 in the paper) into the feature representation dynamics of existing popular scFMs.

Supported backbone models:

| Model | Mode | Key feature |
|-------|------|-------------|
| **scGPT** | Macro-CAL | CLS-token attention contrastive loss |
| **CellFM** | Macro-CAL / MH-CAL | Retention-weight contrastive loss |
| **Geneformer** | MH-CAL + Orthogonal | Per-head contrastive + orthogonality regulariser |

---

## Project Structure

```
CAL/
├── src/
│   ├── cal_loss.py        # CALLoss & MultiHeadCALLoss (Eqs. 8–10)
│   ├── attention_hook.py  # Hook-based attention extractor (Eqs. 1–2)
│   └── __init__.py
├── scripts/
│   ├── train_scgpt.py     # 5-fold CV for scGPT + CAL
│   ├── train_geneformer.py
│   └── train_cellfm.py
├── data/                  # Place .h5ad datasets here (see data/README.md)
├── pretrained_models/     # Place model checkpoints here (see pretrained_models/README.md)
├── results/               # Generated automatically at runtime
├── run_mhcal.sh           # One-click launcher
├── requirements.txt
└── LICENSE
```

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/NTIITM/CAL.git
cd CAL
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download model weights

Follow the instructions in [`pretrained_models/README.md`](pretrained_models/README.md) to download:
- scGPT checkpoint (whole-human pre-trained)
- Geneformer V2-316M weights (from HuggingFace)
- CellFM 80M weights (Google Drive — link released upon paper acceptance)

### 4. Download datasets

Follow the instructions in [`data/README.md`](data/README.md).
The processed `.h5ad` files and Geneformer tokenised datasets will be shared via **Google Drive** upon paper acceptance.

```bash
pip install gdown
# Example:
gdown "https://drive.google.com/uc?id=<FILE_ID>" -O data/hPancreas/hPancreas.h5ad
```

---

## Quick Start

Use the `run_mhcal.sh` one-click launcher:

```bash
# Macro-CAL on scGPT with hPancreas dataset
bash run_mhcal.sh --model scgpt --dataset hPancreas --mode mhcal

# MH-CAL with orthogonal regularisation on Geneformer
bash run_mhcal.sh --model geneformer --dataset hBone --mode mhcal_orth --lambda_orth 0.1

# Baseline (no CAL) for comparison
bash run_mhcal.sh --model cellfm --dataset MS --mode baseline

# Specify GPU and training epochs
bash run_mhcal.sh --model scgpt --dataset hBone --mode mhcal --cuda 2 --epochs 80
```

### Available `--mode` options

| Mode | Description |
|------|-------------|
| `baseline` | Standard cross-entropy fine-tuning, no CAL |
| `ral` | Original RAL (head-averaged attention contrastive) |
| `mhcal` | **MH-CAL** — per-head contrastive loss (recommended) |
| `mhcal_orth` | MH-CAL + head orthogonality regulariser (Eq. 10) |

Results are saved to `results/result_{model}_{dataset}/{mode}/`:
- `kfold_results.json` — per-fold F1 and accuracy
- `kfold_summary.json` — mean ± std across 5 folds

---

## Running Individual Scripts

You can also run the training scripts directly with full control:

```bash
# scGPT
python scripts/train_scgpt.py \
    --dataset hPancreas --mode mhcal --cuda 0 \
    --epochs 80 --batch_size 32 --lambda_attn 0.5

# Geneformer
python scripts/train_geneformer.py \
    --dataset hBone --mode mhcal_orth --cuda 0 \
    --epochs 50 --lambda_orth 0.1

# CellFM
python scripts/train_cellfm.py \
    --dataset MS --mode mhcal --cuda 0 \
    --epochs 10 --batch_size 16
```


## License

This project is released under the [MIT License](LICENSE).

Third-party model implementations (scGPT, CellFM, Geneformer) are subject to their own respective licenses.
