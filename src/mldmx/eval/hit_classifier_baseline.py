"""Evaluation utilities for maintained hit-origin classifier baselines."""

import torch

from mldmx.train.batching import chunks
from mldmx.train.hit_classifier_batching import event_views_from_indices, hit_classifier_batch_kind
from mldmx.train.hit_classifier_baseline import (
    compute_batch_losses,
    compute_event_losses,
    empty_metric_totals,
    finalize_metrics,
    update_metric_totals,
)


@torch.no_grad()
def evaluate(model, events, indices, view_fn, args, device, split_name):
    model.eval()
    batch_kind = hit_classifier_batch_kind(model)
    totals = empty_metric_totals(num_classes=len(args.valid_labels))
    ordered_indices = (
        events.order_indices_for_access(indices)
        if hasattr(events, "order_indices_for_access")
        else indices
    )
    for batch in chunks(ordered_indices, args.batch_size):
        if batch_kind is None:
            for event_idx in batch:
                losses = compute_event_losses(model, events[event_idx], view_fn, device)
                update_metric_totals(totals, losses)
        else:
            views = event_views_from_indices(events, batch, view_fn)
            losses = compute_batch_losses(model, views, batch_kind, device)
            update_metric_totals(totals, losses)
    return finalize_metrics(totals, prefix=f"{split_name}_")
