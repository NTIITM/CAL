#!/bin/bash
# ==============================================================================
# MH-CAL: State-adaptive contrastive attention learning drives mechanistic decoupling in single-cell foundation models
# One-Click Execution Wrapper
# ==============================================================================

set -e

usage() {
    echo "Usage: $0 --model [scgpt|cellfm|geneformer] --dataset [hPancreas|hBone|MS] [options]"
    echo ""
    echo "Options:"
    echo "  --model         Specify the foundation model: scgpt, cellfm, or geneformer"
    echo "  --dataset       Specify the dataset name (e.g., hPancreas)"
    echo "  --mode          Training mode: baseline, ral, mhcal, mhcal_orth (default: mhcal)"
    echo "  --epochs        Number of training epochs (default: 15)"
    echo "  --batch_size    Batch size (default: 64)"
    echo "  --lambda_attn   CAL Contrastive attention weight (default: 0.5)"
    echo "  --lambda_orth   CAL Orthogonal regularization weight (default: 0.0 for scgpt/cellfm, 0.1 for geneformer)"
    echo "  --cuda          GPU device index (default: 0)"
    echo "  --help          Display this help message"
    exit 1
}

# Defaults
MODEL=""
DATASET="hPancreas"
MODE="mhcal"
EPOCHS=15
BATCH=64
L_ATTN=0.5
L_ORTH=""
CUDA_ID=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --batch_size) BATCH="$2"; shift ;;
        --lambda_attn) L_ATTN="$2"; shift ;;
        --lambda_orth) L_ORTH="$2"; shift ;;
        --cuda) CUDA_ID="$2"; shift ;;
        --help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model must be specified."
    usage
fi

if [[ -z "$L_ORTH" ]]; then
    if [[ "$MODEL" == "geneformer" ]]; then
        L_ORTH="0.1" # Geneformer benefits from MH-CAL (orthogonal)
    else
        L_ORTH="0.0" # CellFM and scGPT use macro-CAL
    fi
fi

echo "================================================================="
echo " Starting CAL Pipeline for $MODEL on $DATASET  [mode=$MODE]"
echo " Configuration: Epochs=$EPOCHS, Batch=$BATCH, Lambda_Attn=$L_ATTN, Lambda_Orth=$L_ORTH, CUDA=$CUDA_ID"
echo "="

# Set paths
PROJ_DIR="$(pwd)"
DATA_DIR="$PROJ_DIR/data/$DATASET"
SAVE_DIR="$PROJ_DIR/results/${MODEL}_cal"

mkdir -p "$SAVE_DIR"

if [[ "$MODEL" == "scgpt" ]]; then
    python scripts/train_scgpt.py \
        --mode $MODE \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lambda_attn $L_ATTN \
        --lambda_orth $L_ORTH \
        --cuda $CUDA_ID
elif [[ "$MODEL" == "geneformer" ]]; then
    python scripts/train_geneformer.py \
        --mode $MODE \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lambda_attn $L_ATTN \
        --lambda_orth $L_ORTH \
        --cuda $CUDA_ID
elif [[ "$MODEL" == "cellfm" ]]; then
    python scripts/train_cellfm.py \
        --mode $MODE \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lambda_attn $L_ATTN \
        --cuda $CUDA_ID
else
    echo "Error: Unsupported model $MODEL. Valid choices: scgpt, geneformer, cellfm"
    exit 1
fi

echo "Training complete. Results saved to $SAVE_DIR"
