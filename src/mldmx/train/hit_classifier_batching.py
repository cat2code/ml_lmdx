"""Batch collation for maintained hit-origin classifier baselines."""

from dataclasses import dataclass

import torch


IGNORE_INDEX = -100


@dataclass
class HitClassifierBatch:
    """Batched model input plus token/node-aligned supervision."""

    kind: str
    x: torch.Tensor
    target: torch.Tensor
    supervised_mask: torch.Tensor
    valid_mask: torch.Tensor | None = None
    batch_index: torch.Tensor | None = None
    views: list[dict] | None = None

    @property
    def num_hits(self) -> int:
        return int(self.supervised_mask.sum().item())

    def to(self, device):
        return HitClassifierBatch(
            kind=self.kind,
            x=self.x.to(device=device, dtype=torch.float32),
            target=self.target.to(device=device, dtype=torch.long),
            supervised_mask=self.supervised_mask.to(device=device, dtype=torch.bool),
            valid_mask=(
                None
                if self.valid_mask is None
                else self.valid_mask.to(device=device, dtype=torch.bool)
            ),
            batch_index=(
                None
                if self.batch_index is None
                else self.batch_index.to(device=device, dtype=torch.long)
            ),
            views=self.views,
        )


def hit_classifier_batch_kind(model) -> str | None:
    """Return the supported batching representation for a hit-classifier model."""
    name = model.__class__.__name__
    if "Transformer" in name:
        return "padded"
    if "GravNet" in name:
        return "graph"
    return None


def event_views_from_indices(events_or_views, indices, view_fn):
    """Materialize model-facing views for one index batch."""
    return [
        events_or_views[event_idx] if view_fn is None else view_fn(events_or_views[event_idx])
        for event_idx in indices
    ]


def _target_in_token_space(view, ignore_index=IGNORE_INDEX):
    x = view["x"]
    ecal_mask = view["ecal_mask"].to(dtype=torch.bool)
    target = view["y"].to(dtype=torch.long)
    if x.ndim != 2:
        raise ValueError(f"Expected view['x'] with shape [N, F], got {tuple(x.shape)}.")
    if ecal_mask.shape != (x.shape[0],):
        raise ValueError(
            f"Expected ecal_mask with shape [{x.shape[0]}], got {tuple(ecal_mask.shape)}."
        )
    if target.shape != (int(ecal_mask.sum().item()),):
        raise ValueError(
            "Expected view['y'] to align with supervised ECal hits: "
            f"{tuple(target.shape)} for {int(ecal_mask.sum().item())} ECal hits."
        )

    token_target = torch.full(
        (x.shape[0],),
        int(ignore_index),
        dtype=torch.long,
        device=target.device,
    )
    token_target[ecal_mask.to(device=target.device)] = target
    return token_target, ecal_mask


def collate_transformer_hit_classifier_batch(views, ignore_index=IGNORE_INDEX):
    """
    Collate variable-length events into padded transformer tensors.

    Shapes:
      x: [B, max_tokens, F]
      valid_mask/supervised_mask/target: [B, max_tokens]
    """
    if not views:
        raise ValueError("Cannot collate an empty transformer hit-classifier batch.")

    feature_dim = int(views[0]["x"].shape[1])
    max_tokens = max(int(view["x"].shape[0]) for view in views)
    batch_size = len(views)
    dtype = views[0]["x"].dtype
    x = torch.zeros((batch_size, max_tokens, feature_dim), dtype=dtype)
    target = torch.full((batch_size, max_tokens), int(ignore_index), dtype=torch.long)
    valid_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    supervised_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)

    for row, view in enumerate(views):
        if int(view["x"].shape[1]) != feature_dim:
            raise ValueError("All views in a batch must have the same feature dimension.")
        num_tokens = int(view["x"].shape[0])
        token_target, token_supervised_mask = _target_in_token_space(view, ignore_index)
        x[row, :num_tokens] = view["x"]
        target[row, :num_tokens] = token_target.cpu()
        valid_mask[row, :num_tokens] = True
        supervised_mask[row, :num_tokens] = token_supervised_mask.cpu()

    return HitClassifierBatch(
        kind="padded",
        x=x,
        target=target,
        supervised_mask=supervised_mask,
        valid_mask=valid_mask,
        views=list(views),
    )


def collate_gravnet_hit_classifier_batch(views, ignore_index=IGNORE_INDEX):
    """
    Collate variable-length events into a PyG-style concatenated graph batch.

    Shapes:
      x: [sum_tokens, F]
      batch_index/supervised_mask/target: [sum_tokens]
    """
    if not views:
        raise ValueError("Cannot collate an empty GravNet hit-classifier batch.")

    x_parts = []
    target_parts = []
    supervised_parts = []
    batch_parts = []
    for batch_idx, view in enumerate(views):
        token_target, token_supervised_mask = _target_in_token_space(
            view,
            ignore_index=ignore_index,
        )
        num_tokens = int(view["x"].shape[0])
        x_parts.append(view["x"])
        target_parts.append(token_target.cpu())
        supervised_parts.append(token_supervised_mask.cpu())
        batch_parts.append(torch.full((num_tokens,), batch_idx, dtype=torch.long))

    return HitClassifierBatch(
        kind="graph",
        x=torch.cat(x_parts, dim=0),
        target=torch.cat(target_parts, dim=0),
        supervised_mask=torch.cat(supervised_parts, dim=0),
        batch_index=torch.cat(batch_parts, dim=0),
        views=list(views),
    )


def collate_hit_classifier_batch(views, kind):
    if kind == "padded":
        return collate_transformer_hit_classifier_batch(views)
    if kind == "graph":
        return collate_gravnet_hit_classifier_batch(views)
    raise ValueError(f"Unsupported hit-classifier batch kind: {kind!r}.")
