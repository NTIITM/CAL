# Pretrained Model Weights

This directory contains subdirectories for each supported foundation model.
**Model weights are NOT included in this repository** due to file size.
Please download them manually using the instructions below.

---

## scGPT

**Step 1 — Clone source code:**
```bash
git clone https://github.com/bowang-lab/scGPT.git pretrained_models/scGPT_repo
```

**Step 2 — Download checkpoint:**

Download the scGPT *whole-human* pre-trained model from the
[official scGPT model zoo](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo) and place it at:
```
pretrained_models/scGPT_repo/scgpt/checkpoint/best_model.pt
pretrained_models/scGPT_repo/scgpt/checkpoint/args.json
pretrained_models/scGPT_repo/scgpt/checkpoint/vocab.json
```

> Our experiment checkpoint will be available on Google Drive upon paper acceptance.

---

## Geneformer

Geneformer is available on HuggingFace — download directly:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ctheodoris/Geneformer',
    local_dir='pretrained_models/Geneformer',
    allow_patterns=['config.json', 'model.safetensors', '*.txt', '*.model']
)
"
```

---

## CellFM

**Step 1 — Clone source code** (contact CellFM authors for access):
```bash
# git clone <CellFM_repo_url> pretrained_models/CellFM_torch
```

**Step 2 — Download weights** (place at):
```
pretrained_models/CellFM_weights/CellFM_80M_weight.pth
```

Use `gdown` to download from Google Drive once the link is released:
```bash
pip install gdown
gdown "https://drive.google.com/uc?id=<FILE_ID>" -O pretrained_models/CellFM_weights/CellFM_80M_weight.pth
```

---

## Expected directory structure after setup

```
pretrained_models/
├── scGPT_repo/
│   └── scgpt/checkpoint/
│       ├── best_model.pt
│       ├── args.json
│       └── vocab.json
├── Geneformer/
│   ├── config.json
│   ├── model.safetensors
│   └── ...
├── CellFM_torch/       ← CellFM source code
└── CellFM_weights/
    └── CellFM_80M_weight.pth
```
