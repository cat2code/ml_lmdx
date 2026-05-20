import torch
import torch.nn as nn


class ECalTpadSlotModel(nn.Module):
    """
    MLPF-inspired multi-task model for one LDMX ECal + TriggerPadTracks event.

    The model consumes a variable-length event token tensor with shape
    ``[N_total, F]`` and returns per-node ECal-origin predictions plus event-level
    electron multiplicity predictions. Training code should apply hit-level
    losses only on ECal nodes, using the event's ``ecal_mask`` when available.

    Output class semantics for ``max_electrons=3`` are:
        0 = noise/background
        1 = electron slot 1
        2 = electron slot 2
        3 = electron slot 3

    ``slot_valid_logits[k]`` predicts whether electron slot ``k + 1`` exists in
    the event. ``count_logits`` predicts the total electron count in classes
    ``0..max_electrons``.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_electrons: int = 3,
        dropout: float = 0.0,
        use_type_embedding: bool = False,
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if max_electrons <= 0:
            raise ValueError(f"max_electrons must be positive, got {max_electrons}.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.max_electrons = max_electrons
        self.num_classes = max_electrons + 1
        self.use_type_embedding = use_type_embedding

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.type_embedding = nn.Embedding(2, hidden_dim) if use_type_embedding else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.origin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.fraction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

        self.event_summary = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.slot_valid_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_electrons),
        )
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.signal_head = nn.Linear(hidden_dim, 1)

    def _infer_node_type(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] >= 2:
            is_tpad = x[:, 1] > x[:, 0]
            return is_tpad.to(dtype=torch.long)
        return torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        node_type: torch.Tensor | None = None,
        ecal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: Combined event-token features with shape ``[N_total, F]``.
            node_type: Optional integer tensor with shape ``[N_total]`` where
                ``0`` denotes ECal and ``1`` denotes TriggerPadTracks.
            ecal_mask: Optional ECal-node mask. Accepted for API compatibility;
                the current encoder pools all event tokens.

        Returns:
            Dictionary with:
                ``origin_logits``: ``[N_total, max_electrons + 1]``.
                ``fraction_logits``: ``[N_total, max_electrons + 1]``.
                ``fraction_pred``: softmax of ``fraction_logits``.
                ``slot_valid_logits``: ``[max_electrons]``.
                ``count_logits``: ``[max_electrons + 1]``.
                ``signal_logit``: scalar placeholder for future event labels.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected x with shape [N_total, F], got {tuple(x.shape)}.")
        if x.shape[0] == 0:
            raise ValueError("Expected at least one event token, got an empty tensor.")
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Expected feature dimension {self.in_dim}, got {x.shape[1]}.")

        h = self.input_proj(x)
        if self.type_embedding is not None:
            if node_type is None:
                node_type = self._infer_node_type(x)
            if node_type.shape != (x.shape[0],):
                raise ValueError(
                    f"Expected node_type with shape [{x.shape[0]}], got {tuple(node_type.shape)}."
                )
            h = h + self.type_embedding(node_type.to(device=x.device, dtype=torch.long).clamp(0, 1))

        encoded = self.encoder(h.unsqueeze(0)).squeeze(0)
        mean_repr = encoded.mean(dim=0)
        max_repr = encoded.max(dim=0).values
        log_num_tokens = torch.log1p(
            torch.tensor([encoded.shape[0]], dtype=encoded.dtype, device=encoded.device)
        )
        event_repr = self.event_summary(torch.cat([mean_repr, max_repr, log_num_tokens], dim=0))

        fraction_logits = self.fraction_head(encoded)
        return {
            "origin_logits": self.origin_head(encoded),
            "fraction_logits": fraction_logits,
            "fraction_pred": torch.softmax(fraction_logits, dim=-1),
            "slot_valid_logits": self.slot_valid_head(event_repr),
            "count_logits": self.count_head(event_repr),
            "signal_logit": self.signal_head(event_repr).squeeze(-1),
        }
