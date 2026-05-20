import time

import torch
import torch.nn.functional as F

from mldmx.train.batching import chunks
from mldmx.train.losses import soft_label_cross_entropy
from mldmx.train.progress import make_progress


def ecal_mask_from_event(event: dict) -> torch.Tensor:
    if "ecal_mask" in event:
        return event["ecal_mask"].to(dtype=torch.bool)
    if "num_ecal" in event:
        num_ecal = int(event["num_ecal"])
    elif "y" in event:
        num_ecal = int(event["y"].shape[0])
    else:
        raise KeyError("Event has neither ecal_mask, num_ecal, nor y to identify ECal nodes.")
    mask = torch.zeros((event["x"].shape[0],), dtype=torch.bool)
    mask[:num_ecal] = True
    return mask


def origin_targets_from_event(event: dict, max_electrons: int) -> torch.Tensor:
    if "physical_y" in event:
        target = event["physical_y"].to(dtype=torch.long)
    elif "y" in event:
        target = event["y"].to(dtype=torch.long) + 1
    else:
        raise KeyError("Event is missing both physical_y and y origin targets.")

    if int(target.min().item()) < 0 or int(target.max().item()) > max_electrons:
        raise ValueError(
            f"Origin targets must be in 0..{max_electrons}, got "
            f"{int(target.min().item())}..{int(target.max().item())}."
        )
    return target


def fraction_targets_from_event(
    event: dict,
    origin_target: torch.Tensor,
    max_electrons: int,
) -> torch.Tensor:
    num_classes = max_electrons + 1
    if "fraction_target" not in event:
        return F.one_hot(origin_target.clamp(0, max_electrons), num_classes=num_classes).float()

    fraction_target = event["fraction_target"].to(dtype=torch.float32)
    if fraction_target.shape[1] == num_classes:
        return fraction_target
    if fraction_target.shape[1] == max_electrons:
        noise_column = torch.zeros(
            (fraction_target.shape[0], 1),
            dtype=fraction_target.dtype,
            device=fraction_target.device,
        )
        return torch.cat([noise_column, fraction_target], dim=1)
    raise ValueError(
        f"Expected fraction_target with {max_electrons} or {num_classes} columns, "
        f"got {fraction_target.shape[1]}."
    )


def slot_targets_from_event(
    event: dict,
    origin_target: torch.Tensor,
    fraction_target: torch.Tensor,
    max_electrons: int,
) -> torch.Tensor:
    valid = torch.zeros((max_electrons,), dtype=torch.float32, device=origin_target.device)
    for slot_idx in range(max_electrons):
        class_idx = slot_idx + 1
        # A slot is valid if it owns any hard-label hit or any soft target mass.
        has_hard_hit = bool((origin_target == class_idx).any().item())
        has_fraction_mass = bool((fraction_target[:, class_idx].sum() > 0.0).item())
        valid[slot_idx] = 1.0 if has_hard_hit or has_fraction_mass else 0.0
    return valid


def count_target_from_event(
    event: dict,
    slot_target: torch.Tensor,
    max_electrons: int,
) -> torch.Tensor:
    for key in ("electron_count", "event_electron_count", "count_target", "num_electrons"):
        if key in event:
            value = event[key]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().reshape(-1)[0].item())
            else:
                value = int(value)
            return torch.tensor(min(max(value, 0), max_electrons), dtype=torch.long)
    return slot_target.sum().detach().cpu().to(dtype=torch.long).clamp(max=max_electrons)


def compute_event_losses(model, event: dict, device: torch.device, args):
    x = event["x"].to(device=device, dtype=torch.float32)
    ecal_mask = ecal_mask_from_event(event).to(device)
    outputs = model(x, ecal_mask=ecal_mask)

    origin_target = origin_targets_from_event(event, model.max_electrons).to(device)
    fraction_target = fraction_targets_from_event(
        event,
        origin_target.detach().cpu(),
        model.max_electrons,
    ).to(device)
    slot_target = slot_targets_from_event(
        event,
        origin_target,
        fraction_target,
        model.max_electrons,
    )
    count_target = count_target_from_event(event, slot_target, model.max_electrons).to(device)

    ecal_origin_logits = outputs["origin_logits"][ecal_mask]
    ecal_fraction_logits = outputs["fraction_logits"][ecal_mask]
    ecal_fraction_pred = outputs["fraction_pred"][ecal_mask]

    origin_weight = getattr(args, "origin_class_weights", None)
    if origin_weight is not None:
        origin_weight = torch.as_tensor(origin_weight, dtype=torch.float32, device=device)
    origin_loss = F.cross_entropy(ecal_origin_logits, origin_target, weight=origin_weight)
    fraction_loss = soft_label_cross_entropy(ecal_fraction_logits, fraction_target)
    slot_loss = F.binary_cross_entropy_with_logits(outputs["slot_valid_logits"], slot_target)
    count_weight = getattr(args, "count_class_weights", None)
    if count_weight is not None:
        count_weight = torch.as_tensor(count_weight, dtype=torch.float32, device=device)
    count_loss = F.cross_entropy(
        outputs["count_logits"].unsqueeze(0),
        count_target.unsqueeze(0),
        weight=count_weight,
    )
    total_loss = (
        args.lambda_origin * origin_loss
        + args.lambda_fraction * fraction_loss
        + args.lambda_slot * slot_loss
        + args.lambda_count * count_loss
    )

    pred_class = ecal_origin_logits.argmax(dim=1)
    fraction_abs_error = (ecal_fraction_pred - fraction_target).abs()
    slot_prob = torch.sigmoid(outputs["slot_valid_logits"])
    slot_pred = slot_prob > 0.5
    count_pred = outputs["count_logits"].argmax(dim=-1)
    slot_count_pred = slot_pred.to(dtype=torch.long).sum()

    return {
        "total_loss": total_loss,
        "origin_loss": origin_loss,
        "fraction_loss": fraction_loss,
        "slot_loss": slot_loss,
        "count_loss": count_loss,
        "fraction_mse": F.mse_loss(ecal_fraction_pred, fraction_target),
        "fraction_mae": fraction_abs_error.mean(),
        "per_hit_fraction_mae": fraction_abs_error.mean(dim=1),
        "fraction_target": fraction_target,
        "fraction_pred": ecal_fraction_pred,
        "pred_class": pred_class,
        "true_class": origin_target,
        "slot_target": slot_target,
        "slot_pred": slot_pred,
        "slot_prob": slot_prob,
        "count_target": count_target,
        "count_pred": count_pred,
        "slot_count_pred": slot_count_pred,
        "num_hits": origin_target.numel(),
    }


def empty_slot_metric_totals(num_hit_classes: int, num_count_classes: int):
    return {
        "loss_sum": 0.0,
        "origin_loss_sum": 0.0,
        "fraction_loss_sum": 0.0,
        "slot_loss_sum": 0.0,
        "count_loss_sum": 0.0,
        "fraction_mse_sum": 0.0,
        "fraction_mae_sum": 0.0,
        "correct_hits": 0,
        "hits": 0,
        "events": 0,
        "slot_correct": 0,
        "slot_total": 0,
        "slot_exact_correct": 0,
        "count_correct": 0,
        "slot_count_correct": 0,
        "count_total_by_true": {idx: 0 for idx in range(num_count_classes)},
        "count_correct_by_true": {idx: 0 for idx in range(num_count_classes)},
        "hit_confusion": torch.zeros((num_hit_classes, num_hit_classes), dtype=torch.long),
        "count_confusion": torch.zeros((num_count_classes, num_count_classes), dtype=torch.long),
    }


def update_slot_metric_totals(totals: dict, losses: dict):
    num_hits = int(losses["num_hits"])
    totals["loss_sum"] += float(losses["total_loss"].detach().cpu().item())
    totals["origin_loss_sum"] += float(losses["origin_loss"].detach().cpu().item())
    totals["fraction_loss_sum"] += float(losses["fraction_loss"].detach().cpu().item())
    totals["slot_loss_sum"] += float(losses["slot_loss"].detach().cpu().item())
    totals["count_loss_sum"] += float(losses["count_loss"].detach().cpu().item())
    totals["fraction_mse_sum"] += float(losses["fraction_mse"].detach().cpu().item()) * num_hits
    totals["fraction_mae_sum"] += float(losses["fraction_mae"].detach().cpu().item()) * num_hits

    pred = losses["pred_class"].detach().cpu()
    true = losses["true_class"].detach().cpu()
    totals["correct_hits"] += int((pred == true).sum().item())
    totals["hits"] += num_hits
    for true_idx, pred_idx in zip(true.tolist(), pred.tolist()):
        totals["hit_confusion"][int(true_idx), int(pred_idx)] += 1

    slot_pred = losses["slot_pred"].detach().cpu()
    slot_true = losses["slot_target"].detach().cpu().to(dtype=torch.bool)
    totals["slot_correct"] += int((slot_pred == slot_true).sum().item())
    totals["slot_total"] += int(slot_true.numel())
    totals["slot_exact_correct"] += int(bool((slot_pred == slot_true).all().item()))

    count_true = int(losses["count_target"].detach().cpu().item())
    count_pred = int(losses["count_pred"].detach().cpu().item())
    slot_count_pred = int(losses["slot_count_pred"].detach().cpu().item())
    totals["count_correct"] += int(count_pred == count_true)
    totals["slot_count_correct"] += int(slot_count_pred == count_true)
    totals["count_total_by_true"][count_true] = totals["count_total_by_true"].get(count_true, 0) + 1
    totals["count_correct_by_true"][count_true] = totals["count_correct_by_true"].get(count_true, 0) + int(
        count_pred == count_true
    )
    totals["count_confusion"][count_true, count_pred] += 1
    totals["events"] += 1


def finalize_slot_metrics(totals: dict, prefix: str = ""):
    events = max(1, totals["events"])
    hits = max(1, totals["hits"])
    slot_total = max(1, totals["slot_total"])
    metrics = {
        f"{prefix}loss": totals["loss_sum"] / events,
        f"{prefix}origin_ce": totals["origin_loss_sum"] / events,
        f"{prefix}fraction_ce": totals["fraction_loss_sum"] / events,
        f"{prefix}slot_bce": totals["slot_loss_sum"] / events,
        f"{prefix}count_ce": totals["count_loss_sum"] / events,
        f"{prefix}fraction_mse": totals["fraction_mse_sum"] / hits,
        f"{prefix}fraction_mae": totals["fraction_mae_sum"] / hits,
        f"{prefix}accuracy": totals["correct_hits"] / hits,
        f"{prefix}slot_accuracy": totals["slot_correct"] / slot_total,
        f"{prefix}slot_exact_accuracy": totals["slot_exact_correct"] / events,
        f"{prefix}count_accuracy": totals["count_correct"] / events,
        f"{prefix}slot_count_accuracy": totals["slot_count_correct"] / events,
        f"{prefix}num_hits": totals["hits"],
        f"{prefix}num_events": totals["events"],
    }
    for count, total in sorted(totals["count_total_by_true"].items()):
        if total > 0:
            metrics[f"{prefix}count_accuracy_{count}e"] = (
                totals["count_correct_by_true"].get(count, 0) / total
            )
    return metrics


def event_prediction_record(event_idx: int, event: dict, losses: dict) -> dict:
    raw_event_id = event.get("event_idx", event_idx)
    if isinstance(raw_event_id, torch.Tensor):
        raw_event_id = int(raw_event_id.detach().cpu().reshape(-1)[0].item())
    record = {
        "event_index": int(event_idx),
        "event_id": int(raw_event_id),
        "true_count": int(losses["count_target"].detach().cpu().item()),
        "predicted_count": int(losses["count_pred"].detach().cpu().item()),
        "predicted_slot_count": int(losses["slot_count_pred"].detach().cpu().item()),
        "slot_target": [float(value) for value in losses["slot_target"].detach().cpu().tolist()],
        "slot_probability": [float(value) for value in losses["slot_prob"].detach().cpu().tolist()],
    }
    for key in ("source_file", "source_entry", "source_label"):
        if key in event:
            value = event[key]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().reshape(-1)[0].item())
            record[key] = value
    return record


def train_one_epoch(model, events, train_indices, optimizer, args, device, epoch, logger):
    model.train()
    generator = torch.Generator().manual_seed(args.seed + epoch)
    shuffled_indices = [
        train_indices[idx]
        for idx in torch.randperm(len(train_indices), generator=generator).tolist()
    ]
    batch_indices = list(chunks(shuffled_indices, args.batch_size))
    totals = empty_slot_metric_totals(
        num_hit_classes=args.max_electrons + 1,
        num_count_classes=args.max_electrons + 1,
    )
    start_time = time.time()
    progress = make_progress(
        batch_indices,
        total=len(batch_indices),
        desc=f"epoch {epoch + 1}/{args.epochs} train",
        disable=args.no_progress,
        unit="batch",
    )

    for batch in progress:
        optimizer.zero_grad(set_to_none=True)
        batch_loss_sum = None
        for event_idx in batch:
            losses = compute_event_losses(model, events[event_idx], device, args)
            update_slot_metric_totals(totals, losses)
            batch_loss_sum = losses["total_loss"] if batch_loss_sum is None else batch_loss_sum + losses["total_loss"]

        batch_loss = batch_loss_sum / max(1, len(batch))
        batch_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if hasattr(progress, "set_postfix"):
            metrics = finalize_slot_metrics(totals)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                hit_acc=f"{metrics['accuracy']:.3f}",
                count_acc=f"{metrics['count_accuracy']:.3f}",
            )

    metrics = finalize_slot_metrics(totals, prefix="train_")
    metrics["train_elapsed_sec"] = time.time() - start_time
    logger.info(
        (
            "epoch=%03d train_loss=%.5f train_hit_acc=%.4f "
            "train_count_acc=%.4f train_fraction_mae=%.5f elapsed=%.1fs"
        ),
        epoch + 1,
        metrics["train_loss"],
        metrics["train_accuracy"],
        metrics["train_count_accuracy"],
        metrics["train_fraction_mae"],
        metrics["train_elapsed_sec"],
    )
    return metrics
