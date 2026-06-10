"""Maintained full-self-attention hit-origin classification baselines."""

import torch
import torch.nn as nn


class _TransformerHitClassifier(nn.Module):
    """Per-token classifier using full self-attention over one event."""

    def __init__(
        self,
        in_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        out_dim: int = 3,
    ):
        super().__init__()
        if in_dim <= 0 or d_model <= 0 or out_dim <= 0:
            raise ValueError("in_dim, d_model, and out_dim must be positive.")
        if nhead <= 0 or num_layers <= 0 or dim_feedforward <= 0:
            raise ValueError("nhead, num_layers, and dim_feedforward must be positive.")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Return origin-class logits for every input token.

        Args:
            x: One event ``[N_tokens, in_dim]`` or a padded batch
                ``[B, max_tokens, in_dim]``.
            key_padding_mask: Optional bool mask with shape ``[B, max_tokens]``
                where True entries are padding tokens ignored by attention.
        """
        if x.ndim not in (2, 3) or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected x with shape [N_tokens, {self.in_dim}] or "
                f"[B, N_tokens, {self.in_dim}], got {tuple(x.shape)}."
            )
        if x.shape[-2] == 0:
            raise ValueError("Expected at least one token per transformer batch.")

        single_event = x.ndim == 2
        if single_event:
            x = x.unsqueeze(0)
            if key_padding_mask is not None and key_padding_mask.ndim == 1:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        if key_padding_mask is not None:
            expected_mask_shape = x.shape[:2]
            if key_padding_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"Expected key_padding_mask with shape {tuple(expected_mask_shape)}, "
                    f"got {tuple(key_padding_mask.shape)}."
                )
            key_padding_mask = key_padding_mask.to(device=x.device, dtype=torch.bool)

        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        logits = self.head(h)
        return logits.squeeze(0) if single_event else logits


class ECalTransformer(_TransformerHitClassifier):
    """ECal-only full-self-attention hit-origin classifier."""


class ECalTpadTransformer(_TransformerHitClassifier):
    """ECal plus TriggerPadTracks full-self-attention hit-origin classifier."""


class ECalHitTransformer(ECalTransformer):
    """Compatibility class for older prototype scripts."""
