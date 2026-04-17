"""
CAL (Contrastive Attention Learning) — core algorithm package.

Main exports:
    CALLoss          — single-head contrastive attention loss with memory queue
    MultiHeadCALLoss — per-head contrastive loss with optional orthogonality regularizer
    AttentionExtractor — hook-based attention extractor for scGPT-style models
"""

from .cal_loss import CALLoss, MultiHeadCALLoss
from .attention_hook import AttentionExtractor

__all__ = ["CALLoss", "MultiHeadCALLoss", "AttentionExtractor"]
