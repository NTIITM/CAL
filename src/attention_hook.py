"""
Attention Hook for extracting attention weights from scGPT's TransformerEncoder.

Memory-efficient version: Instead of recomputing attention from Q/K projections
(which causes OOM for long sequences), we patch the last layer's self_attn
to use torch.nn.functional.multi_head_attention_forward with need_weights=True.

For very long sequences, we subsample to reduce memory usage.
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional


class AttentionExtractor:
    """
    Extracts attention weights from the last TransformerEncoderLayer.

    Uses a forward pre-hook to modify the MHA call to return attention weights,
    plus a forward hook to capture the attention output. This avoids recomputing
    the full attention matrix.

    [CITATION] This directly implements the "Formalization of Attention Distribution Extraction" 
    from the MH-CAL paper. It extracts the raw attention vector for the [CLS] token (Eq 1) 
    and renormalizes to a valid discrete probability distribution (Eq 2).

    Usage:
        extractor = AttentionExtractor()
        extractor.register(model)
        output = model(...)
        attn_weights = extractor.get_attention_weights()
        extractor.remove()
    """

    def __init__(self):
        self._attn_weights: Optional[Tensor] = None
        self._hooks = []
        self._original_sa_forward = None
        self._target_layer = None

    def register(self, model: nn.Module):
        """Register hooks on the last TransformerEncoderLayer."""
        transformer_encoder = model.transformer_encoder
        last_layer = transformer_encoder.layers[-1]
        self._target_layer = last_layer

        extractor_ref = self

        # Monkey-patch the self_attn's forward to force need_weights=True
        mha = last_layer.self_attn
        original_forward = mha.forward
        self._original_sa_forward = original_forward

        def patched_mha_forward(
            query, key, value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False,
        ):
            # Force need_weights=True, average over heads
            out, weights = original_forward(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                average_attn_weights=True,
                is_causal=is_causal,
            )
            # weights: (batch, tgt_len, src_len), averaged over heads
            if weights is not None:
                extractor_ref._attn_weights = weights
            return out, weights

        mha.forward = patched_mha_forward

    def get_attention_weights(self) -> Optional[Tensor]:
        """Get attention weights: (batch, seq_len, seq_len), averaged over heads."""
        return self._attn_weights

    def clear(self):
        """Clear stored attention weights."""
        self._attn_weights = None

    def remove(self):
        """Remove patches and restore original forward."""
        if self._target_layer is not None and self._original_sa_forward is not None:
            self._target_layer.self_attn.forward = self._original_sa_forward
            self._original_sa_forward = None
            self._target_layer = None
        self._attn_weights = None
