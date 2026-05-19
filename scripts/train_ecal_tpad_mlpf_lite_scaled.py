"""
Scaled ECAL/TPAD MLPF-lite training script.

Example from the repository root:

    python mldmx/scripts/train_ecal_tpad_mlpf_lite_scaled.py --max-events 1000 --epochs 1

Example from the mldmx directory:

    pip install -e .; python scripts/train_ecal_tpad_mlpf_lite_scaled.py --max-events 15000 --epochs 20

    pip install -e .; python scripts/train_ecal_tpad_mlpf_lite_scaled.py --max-events 4000 --epochs 20
"""

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised only when tqdm is missing.
    tqdm = None

from mldmx.datasets.ecal_tpad_loading import load_ecal_tpad_tensor_events
from mldmx.io.root_files import find_root_files, root_file_sort_key
from mldmx.models import ECalTpadMLPFLiteTransformer
from mldmx.train.losses import soft_label_cross_entropy
from mldmx.train.utils import choose_device
from mldmx.viz.ecal import plot_ecal_hit_classes_3d


VALID_LABELS = (1, 2, 3)
DEFAULT_DATA_DIR = Path("data/ldmx_overlay_events_700k/3e/events")


class SimpleProgress:
    def __init__(self, iterable, total=None, desc="", unit="item"):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.unit = unit

    def __iter__(self):
        for idx, item in enumerate(self.iterable, start=1):
            if self.total and (idx == 1 or idx == self.total or idx % max(1, self.total // 10) == 0):
                print(f"{self.desc}: {self.unit} {idx}/{self.total}")
            yield item

    def set_postfix(self, **_kwargs):
        return None


def make_progress(iterable, total=None, desc="", disable=False, unit="event"):
    if disable:
        return iterable
    if tqdm is None:
        return SimpleProgress(iterable, total=total, desc=desc, unit=unit)
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False, unit=unit)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train the ECAL/TPAD MLPF-lite transformer on a configurable ROOT event subset."
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs/ecal_tpad_mlpf_lite_scaled",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory name under --output-dir. Defaults to a timestamped run name.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of events per optimizer step.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Defaults to CUDA when available, then MPS, then CPU.",
    )
    parser.add_argument("--valid-labels", type=int, nargs="+", default=list(VALID_LABELS))
    parser.add_argument(
        "--target-mode",
        choices=("canonical-y", "canonical-x", "canonical-z", "physical-origin"),
        default="canonical-y",
        help=(
            "physical-origin trains on raw origin_id labels. canonical-* remaps each event's "
            "origins into a deterministic spatial order, which avoids arbitrary label "
            "permutations between otherwise identical 3-electron events."
        ),
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lambda-fraction", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument("--keep-noise", action="store_true")
    parser.add_argument("--num-plot-hits", type=int, default=20000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--event-log-every", type=int, default=0)
    parser.add_argument(
        "--read-step-size",
        type=int,
        default=500,
        help=(
            "Number of ROOT entries per chunk while reading. Smaller values update loading "
            "progress more often; larger values reduce iterator overhead."
        ),
    )
    parser.add_argument(
        "--allow-fewer-events",
        action="store_true",
        help="Continue if fewer events than --max-events are found.",
    )
    return parser.parse_args()


def setup_logging(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    logger = logging.getLogger("ecal_tpad_mlpf_lite_scaled")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def resolve_run_dir(args):
    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return args.output_dir / run_name


def resolve_data_dir(data_dir):
    if data_dir.exists():
        return data_dir

    script_project_root = Path(__file__).resolve().parents[1]
    candidates = [
        script_project_root / data_dir,
        script_project_root.parent / data_dir,
        script_project_root / "data/ldmx_overlay_events_700k/3e/events",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find data directory '{data_dir}'. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_tensor_events(args, data_dir, root_files, logger):
    filter_noise = not args.keep_noise
    read_step_size = args.read_step_size if args.read_step_size > 0 else None
    return load_ecal_tpad_tensor_events(
        root_files=root_files,
        max_events=args.max_events,
        valid_labels=tuple(args.valid_labels),
        target_mode=args.target_mode,
        filter_noise=filter_noise,
        allow_fewer_events=args.allow_fewer_events,
        data_dir=data_dir,
        logger=logger,
        progress_factory=make_progress,
        disable_progress=args.no_progress,
        event_log_every=args.event_log_every,
        read_step_size=read_step_size,
    )


def deterministic_split(num_events, seed):
    if num_events < 20:
        raise ValueError(
            f"Need at least 20 events for an 80/15/5 split with a non-empty test set; got {num_events}."
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_events, generator=generator).tolist()
    n_train = int(0.80 * num_events)
    n_val = int(0.15 * num_events)
    n_test = num_events - n_train - n_val
    if n_val == 0 or n_test == 0:
        raise ValueError(f"Split produced empty validation/test sets for {num_events} events.")
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def count_classes(events, indices):
    counter = Counter()
    for idx in indices:
        counter.update(events[idx]["physical_y"].tolist())
    return dict(sorted(counter.items()))


def target_order_counts(events, indices):
    counter = Counter()
    for idx in indices:
        order = events[idx].get("target_label_order")
        if order is not None:
            counter.update([tuple(order)])
    return {str(key): value for key, value in sorted(counter.items())}


def normalize_continuous_features(tensor_events, train_indices, first_continuous_col=2):
    train_x = torch.cat(
        [tensor_events[idx]["x"][:, first_continuous_col:] for idx in train_indices],
        dim=0,
    )
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(1e-6)

    for event in tensor_events:
        x = event["x"].clone()
        x[:, first_continuous_col:] = (x[:, first_continuous_col:] - mean) / std
        event["x"] = x

    return {
        "first_continuous_col": first_continuous_col,
        "mean": mean,
        "std": std,
    }


def resolve_device(requested_device, logger):
    cuda_available = torch.cuda.is_available()
    logger.info("CUDA available: %s", cuda_available)
    if cuda_available:
        logger.info("CUDA GPU: %s", torch.cuda.get_device_name(0))

    if requested_device == "cuda" and not cuda_available:
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested_device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return choose_device(requested_device)


def count_trainable_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def model_kwargs_from_args(args, input_dim):
    return {
        "input_dim": input_dim,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "out_dim": len(args.valid_labels),
    }


def chunks(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def compute_event_losses(model, event, device, lambda_fraction):
    x = event["x"].to(device)
    ecal_mask = event["ecal_mask"].to(device)
    y = event["y"].to(device)
    fraction_target = event["fraction_target"].to(device)

    outputs = model(x)
    ecal_origin_logits = outputs["origin_logits"][ecal_mask]
    ecal_fraction_logits = outputs["fraction_logits"][ecal_mask]
    ecal_fraction_pred = outputs["fraction_pred"][ecal_mask]

    origin_loss = F.cross_entropy(ecal_origin_logits, y)
    fraction_loss = soft_label_cross_entropy(ecal_fraction_logits, fraction_target)
    total_loss = origin_loss + lambda_fraction * fraction_loss

    pred_class = ecal_origin_logits.argmax(dim=1)
    fraction_abs_error = (ecal_fraction_pred - fraction_target).abs()
    return {
        "total_loss": total_loss,
        "origin_loss": origin_loss,
        "fraction_loss": fraction_loss,
        "fraction_mse": F.mse_loss(ecal_fraction_pred, fraction_target),
        "fraction_mae": fraction_abs_error.mean(),
        "per_hit_fraction_mae": fraction_abs_error.mean(dim=1),
        "fraction_target": fraction_target,
        "fraction_pred": ecal_fraction_pred,
        "pred_class": pred_class,
        "true_class": y,
        "num_hits": y.numel(),
    }


def empty_metric_totals(num_classes):
    return {
        "loss_sum": 0.0,
        "origin_loss_sum": 0.0,
        "fraction_loss_sum": 0.0,
        "fraction_mse_sum": 0.0,
        "fraction_mae_sum": 0.0,
        "correct": 0,
        "hits": 0,
        "confusion": torch.zeros((num_classes, num_classes), dtype=torch.long),
    }


def update_metric_totals(totals, losses):
    num_hits = int(losses["num_hits"])
    totals["loss_sum"] += float(losses["total_loss"].detach().cpu().item()) * num_hits
    totals["origin_loss_sum"] += float(losses["origin_loss"].detach().cpu().item()) * num_hits
    totals["fraction_loss_sum"] += float(losses["fraction_loss"].detach().cpu().item()) * num_hits
    totals["fraction_mse_sum"] += float(losses["fraction_mse"].detach().cpu().item()) * num_hits
    totals["fraction_mae_sum"] += float(losses["fraction_mae"].detach().cpu().item()) * num_hits
    pred = losses["pred_class"].detach().cpu()
    true = losses["true_class"].detach().cpu()
    totals["correct"] += int((pred == true).sum().item())
    totals["hits"] += num_hits
    for true_idx, pred_idx in zip(true.tolist(), pred.tolist()):
        totals["confusion"][int(true_idx), int(pred_idx)] += 1


def finalize_metrics(totals, prefix=""):
    hits = max(1, totals["hits"])
    metrics = {
        f"{prefix}loss": totals["loss_sum"] / hits,
        f"{prefix}origin_ce": totals["origin_loss_sum"] / hits,
        f"{prefix}fraction_ce": totals["fraction_loss_sum"] / hits,
        f"{prefix}fraction_mse": totals["fraction_mse_sum"] / hits,
        f"{prefix}fraction_mae": totals["fraction_mae_sum"] / hits,
        f"{prefix}accuracy": totals["correct"] / hits,
        f"{prefix}num_hits": totals["hits"],
    }
    class_metrics = classification_metrics_from_confusion(totals["confusion"])
    for key, value in class_metrics.items():
        metrics[f"{prefix}{key}"] = value
    return metrics


def classification_metrics_from_confusion(confusion):
    confusion = confusion.to(dtype=torch.float64)
    tp = confusion.diag()
    pred_count = confusion.sum(dim=0).clamp_min(1.0)
    true_count = confusion.sum(dim=1).clamp_min(1.0)
    precision = tp / pred_count
    recall = tp / true_count
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    weights = true_count / true_count.sum().clamp_min(1.0)
    return {
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "weighted_f1": float((f1 * weights).sum().item()),
    }


def train_one_epoch(model, events, train_indices, optimizer, args, device, epoch, logger):
    model.train()
    generator = torch.Generator().manual_seed(args.seed + epoch)
    shuffled_indices = [
        train_indices[idx]
        for idx in torch.randperm(len(train_indices), generator=generator).tolist()
    ]
    batch_indices = list(chunks(shuffled_indices, args.batch_size))
    totals = empty_metric_totals(num_classes=len(args.valid_labels))
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
        batch_hits = 0
        for event_idx in batch:
            losses = compute_event_losses(model, events[event_idx], device, args.lambda_fraction)
            update_metric_totals(totals, losses)
            weighted_loss = losses["total_loss"] * losses["num_hits"]
            batch_loss_sum = weighted_loss if batch_loss_sum is None else batch_loss_sum + weighted_loss
            batch_hits += int(losses["num_hits"])

        batch_loss = batch_loss_sum / max(1, batch_hits)
        batch_loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if hasattr(progress, "set_postfix"):
            metrics = finalize_metrics(totals)
            progress.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                acc=f"{metrics['accuracy']:.3f}",
            )

    metrics = finalize_metrics(totals, prefix="train_")
    metrics["train_elapsed_sec"] = time.time() - start_time
    logger.info(
        "epoch=%03d train_loss=%.5f train_acc=%.4f train_fraction_mae=%.5f elapsed=%.1fs",
        epoch + 1,
        metrics["train_loss"],
        metrics["train_accuracy"],
        metrics["train_fraction_mae"],
        metrics["train_elapsed_sec"],
    )
    return metrics


@torch.no_grad()
def evaluate(model, events, indices, args, device, split_name, collect_plot_samples=False):
    model.eval()
    totals = empty_metric_totals(num_classes=len(args.valid_labels))
    plot_samples = {
        "fraction_target": [],
        "fraction_pred": [],
        "per_hit_fraction_mae": [],
    }
    remaining_plot_hits = args.num_plot_hits if collect_plot_samples else 0

    for batch in chunks(indices, args.batch_size):
        for event_idx in batch:
            losses = compute_event_losses(model, events[event_idx], device, args.lambda_fraction)
            update_metric_totals(totals, losses)

            if remaining_plot_hits > 0:
                take = min(remaining_plot_hits, int(losses["num_hits"]))
                plot_samples["fraction_target"].append(losses["fraction_target"][:take].detach().cpu())
                plot_samples["fraction_pred"].append(losses["fraction_pred"][:take].detach().cpu())
                plot_samples["per_hit_fraction_mae"].append(
                    losses["per_hit_fraction_mae"][:take].detach().cpu()
                )
                remaining_plot_hits -= take

    metrics = finalize_metrics(totals, prefix=f"{split_name}_")
    metrics[f"{split_name}_confusion"] = totals["confusion"].tolist()

    if collect_plot_samples:
        for key, tensors in plot_samples.items():
            plot_samples[key] = torch.cat(tensors, dim=0) if tensors else torch.empty((0,))
        return metrics, plot_samples
    return metrics, None


def checkpoint_state(model, optimizer, scheduler, epoch, args, history, best_val_loss, model_kwargs, feature_norm, splits):
    return {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "history": history,
        "args": vars(args),
        "best_val_loss": best_val_loss,
        "model_kwargs": model_kwargs,
        "feature_norm": {
            "first_continuous_col": feature_norm["first_continuous_col"],
            "mean": feature_norm["mean"].detach().cpu().tolist(),
            "std": feature_norm["std"].detach().cpu().tolist(),
        }
        if feature_norm is not None
        else None,
        "splits": splits,
        "valid_labels": tuple(args.valid_labels),
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, args, history, best_val_loss, model_kwargs, feature_norm, splits):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_state(
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            history,
            best_val_loss,
            model_kwargs,
            feature_norm,
            splits,
        ),
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_history(history, run_dir):
    save_json(run_dir / "history.json", history)
    csv_path = run_dir / "history.csv"
    if not history:
        return
    fieldnames = sorted({key for row in history for key in row.keys() if not key.endswith("_confusion")})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_config(args, run_dir, data_dir, root_files, event_sources, splits, target_order_counts_by_split):
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload.update(
        {
            "resolved_data_dir": str(data_dir),
            "root_files_used": sorted(
                {source["file"] for source in event_sources},
                key=lambda name: root_file_sort_key(Path(name)),
            ),
            "num_loaded_events": len(event_sources),
            "split_sizes": {key: len(value) for key, value in splits.items()},
            "target_order_counts": target_order_counts_by_split,
        }
    )
    save_json(run_dir / "config.json", payload)


def plot_history(history, run_dir):
    if not history:
        return
    epochs = [row["epoch"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    ax.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("ECAL/TPAD MLPF-lite loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "loss_history.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    ax.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("ECAL/TPAD MLPF-lite accuracy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy_history.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(confusion, valid_labels, output_path, title):
    confusion = torch.as_tensor(confusion, dtype=torch.float64)
    row_sums = confusion.sum(dim=1, keepdim=True).clamp_min(1.0)
    normalized = confusion / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalized.numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_xticks(range(len(valid_labels)), labels=[str(label) for label in valid_labels])
    ax.set_yticks(range(len(valid_labels)), labels=[str(label) for label in valid_labels])

    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            count = int(confusion[row, col].item())
            frac = normalized[row, col].item()
            text_color = "white" if frac > 0.5 else "black"
            ax.text(col, row, f"{count}\n{frac:.2f}", ha="center", va="center", color=text_color)

    fig.colorbar(image, ax=ax, label="row-normalized fraction")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_test_fraction_summaries(plot_samples, args, run_dir):
    if plot_samples is None or plot_samples["fraction_target"].numel() == 0:
        return

    target = plot_samples["fraction_target"].numpy()
    pred = plot_samples["fraction_pred"].numpy()
    valid_labels = tuple(args.valid_labels)

    fig, axes = plt.subplots(1, len(valid_labels), figsize=(4 * len(valid_labels), 4), sharex=True, sharey=True)
    axes = axes if isinstance(axes, (list, tuple)) else getattr(axes, "flat", [axes])
    for idx, (ax, origin) in enumerate(zip(axes, valid_labels)):
        ax.scatter(target[:, idx], pred[:, idx], s=5, alpha=0.35)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_title(f"origin {origin}")
        ax.set_xlabel("true fraction")
        if idx == 0:
            ax.set_ylabel("predicted fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Test-set origin energy fractions")
    fig.tight_layout()
    fig.savefig(run_dir / "test_fraction_scatter.png", dpi=200)
    plt.close(fig)

    target_max = target.max(axis=1)
    pred_max = pred.max(axis=1)
    bins = torch.linspace(0, 1, 31).numpy()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(target_max, bins=bins, alpha=0.55, label="true max fraction")
    ax.hist(pred_max, bins=bins, alpha=0.55, label="predicted max fraction")
    ax.set_xlabel("max origin fraction per ECal hit")
    ax.set_ylabel("hits")
    ax.set_title("Test-set fraction purity")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "test_fraction_purity.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(plot_samples["per_hit_fraction_mae"].numpy(), bins=30, alpha=0.75)
    ax.set_xlabel("mean absolute fraction error per ECal hit")
    ax.set_ylabel("hits")
    ax.set_title("Test-set fraction MAE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(run_dir / "test_fraction_mae_hist.png", dpi=200)
    plt.close(fig)


@torch.no_grad()
def plot_test_ecal_hit_classes(model, events, test_indices, args, device, run_dir):
    if not test_indices:
        return

    event_idx = test_indices[0]
    event = events[event_idx]
    x = event["x"].to(device)
    ecal_mask = event["ecal_mask"].to(device)
    outputs = model(x)
    pred_class = outputs["origin_logits"][ecal_mask].argmax(dim=1).detach().cpu()

    if args.target_mode == "physical-origin":
        class_to_label = {
            class_idx: label
            for class_idx, label in enumerate(tuple(args.valid_labels))
        }
        pred_labels = torch.tensor(
            [class_to_label[int(class_idx)] for class_idx in pred_class.tolist()],
            dtype=torch.long,
        )
        label_name = "origin_id"
    else:
        pred_labels = pred_class + 1
        label_name = f"{args.target_mode} target class"

    pos = event["ecal_pos"].detach().cpu()
    true_labels = event["physical_y"].detach().cpu()
    plot_ecal_hit_classes_3d(
        pos,
        true_labels,
        run_dir / "test_ecal_classes_truth.png",
        f"Test event {event_idx} ECal hits, true {label_name}",
    )
    plot_ecal_hit_classes_3d(
        pos,
        pred_labels,
        run_dir / "test_ecal_classes_predicted.png",
        f"Test event {event_idx} ECal hits, predicted {label_name}",
    )


def make_scheduler(optimizer, args):
    if args.lr_scheduler == "none":
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
    )


def validate_args(args):
    if args.max_events <= 0:
        raise ValueError("--max-events must be positive.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    if args.read_step_size < 0:
        raise ValueError("--read-step-size must be non-negative.")
    valid_labels = tuple(args.valid_labels)
    if len(valid_labels) < 2:
        raise ValueError(f"--valid-labels must contain at least two labels, got {valid_labels}.")
    if len(set(valid_labels)) != len(valid_labels):
        raise ValueError(f"--valid-labels contains duplicates: {valid_labels}.")


def main():
    args = parse_args()
    validate_args(args)
    run_dir = resolve_run_dir(args)
    logger = setup_logging(run_dir)
    torch.manual_seed(args.seed)

    data_dir = resolve_data_dir(args.data_dir)
    root_files = find_root_files(data_dir)
    logger.info("Output directory: %s", run_dir)
    logger.info("Data directory: %s", data_dir)
    logger.info("First ROOT files: %s", ", ".join(path.name for path in root_files[:5]))
    logger.info("Noise handling: %s", "keeping noise hits" if args.keep_noise else "filtering noise hits")

    device = resolve_device(args.device, logger)
    logger.info("Using device: %s", device)

    events, event_sources = load_tensor_events(args, data_dir, root_files, logger)
    splits = deterministic_split(len(events), args.seed)
    logger.info(
        "Split sizes: train=%s val=%s test=%s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    logger.info("Target mode: %s", args.target_mode)
    if args.target_mode != "physical-origin":
        logger.info(
            "Training origin-id order counts after canonicalization: %s",
            target_order_counts(events, splits["train"]),
        )
    logger.info("Training class counts: %s", count_classes(events, splits["train"]))
    logger.info("Validation class counts: %s", count_classes(events, splits["val"]))
    logger.info("Test class counts: %s", count_classes(events, splits["test"]))

    feature_norm = None
    if not args.no_normalize_features:
        feature_norm = normalize_continuous_features(events, splits["train"])
        logger.info("Normalized continuous feature columns from training split statistics.")

    input_dim = events[0]["x"].shape[1]
    model_kwargs = model_kwargs_from_args(args, input_dim=input_dim)
    model = ECalTpadMLPFLiteTransformer(**model_kwargs).to(device)
    logger.info("Trainable model parameters: %s", count_trainable_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args)

    history = []
    best_val_loss = float("inf")
    start_epoch = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        checkpoint_splits = checkpoint.get("splits")
        if checkpoint_splits is not None and checkpoint_splits != splits:
            raise ValueError(
                "The requested run split does not match the checkpoint split. "
                "Resume with the same --max-events and --seed used for the checkpoint."
            )
        checkpoint_labels = tuple(checkpoint.get("valid_labels", ()))
        if checkpoint_labels and checkpoint_labels != tuple(args.valid_labels):
            raise ValueError(
                f"Checkpoint valid labels {checkpoint_labels} do not match current labels "
                f"{tuple(args.valid_labels)}."
            )
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_target_mode = checkpoint_args.get("target_mode")
        if checkpoint_target_mode is not None and checkpoint_target_mode != args.target_mode:
            raise ValueError(
                f"Checkpoint target mode {checkpoint_target_mode!r} does not match current "
                f"target mode {args.target_mode!r}."
            )
        history = checkpoint.get("history", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = int(checkpoint["epoch"]) + 1
        logger.info("Resumed from %s at next epoch %s", args.resume, start_epoch + 1)

    save_config(
        args,
        run_dir,
        data_dir,
        root_files,
        event_sources,
        splits,
        {split_name: target_order_counts(events, indices) for split_name, indices in splits.items()},
    )
    save_history(history, run_dir)

    interrupted = False
    try:
        for epoch in range(start_epoch, args.epochs):
            lr = optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "epoch": epoch + 1,
                "lr": lr,
            }
            train_metrics = train_one_epoch(model, events, splits["train"], optimizer, args, device, epoch, logger)
            val_start = time.time()
            val_metrics, _ = evaluate(model, events, splits["val"], args, device, "val")
            val_metrics["val_elapsed_sec"] = time.time() - val_start
            epoch_metrics.update(train_metrics)
            epoch_metrics.update(val_metrics)
            history.append(epoch_metrics)

            logger.info(
                "epoch=%03d val_loss=%.5f val_acc=%.4f val_macro_f1=%.4f lr=%.3g",
                epoch + 1,
                val_metrics["val_loss"],
                val_metrics["val_accuracy"],
                val_metrics["val_macro_f1"],
                lr,
            )

            if scheduler is not None:
                scheduler.step(val_metrics["val_loss"])

            save_history(history, run_dir)
            plot_history(history, run_dir)

            latest_path = run_dir / "checkpoints/latest.pt"
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                history,
                best_val_loss,
                model_kwargs,
                feature_norm,
                splits,
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                save_checkpoint(
                    run_dir / "checkpoints/best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    history,
                    best_val_loss,
                    model_kwargs,
                    feature_norm,
                    splits,
                )
                logger.info("Saved new best checkpoint with val_loss=%.5f", best_val_loss)

            if (epoch + 1) % args.checkpoint_every == 0:
                save_checkpoint(
                    run_dir / f"checkpoints/epoch_{epoch + 1:04d}.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    history,
                    best_val_loss,
                    model_kwargs,
                    feature_norm,
                    splits,
                )

    except KeyboardInterrupt:
        interrupted = True
        logger.warning("KeyboardInterrupt received; saving partial run artifacts.")
        save_checkpoint(
            run_dir / "checkpoints/interrupted_latest.pt",
            model,
            optimizer,
            scheduler,
            max(start_epoch - 1, len(history) - 1),
            args,
            history,
            best_val_loss,
            model_kwargs,
            feature_norm,
            splits,
        )

    finally:
        save_history(history, run_dir)
        plot_history(history, run_dir)
        try:
            test_start = time.time()
            test_metrics, plot_samples = evaluate(
                model,
                events,
                splits["test"],
                args,
                device,
                "test",
                collect_plot_samples=True,
            )
            test_metrics["test_elapsed_sec"] = time.time() - test_start
            val_metrics, _ = evaluate(model, events, splits["val"], args, device, "final_val")
            final_metrics = {
                "interrupted": interrupted,
                "best_val_loss": best_val_loss,
                **val_metrics,
                **test_metrics,
            }
            save_json(run_dir / "final_metrics.json", final_metrics)
            plot_confusion_matrix(
                test_metrics["test_confusion"],
                tuple(args.valid_labels),
                run_dir / "test_confusion_matrix.png",
                "Test confusion matrix",
            )
            plot_test_fraction_summaries(plot_samples, args, run_dir)
            plot_test_ecal_hit_classes(model, events, splits["test"], args, device, run_dir)
            save_checkpoint(
                run_dir / "checkpoints/latest.pt",
                model,
                optimizer,
                scheduler,
                max(start_epoch - 1, len(history) - 1),
                args,
                history,
                best_val_loss,
                model_kwargs,
                feature_norm,
                splits,
            )
            logger.info(
                "Final test: loss=%.5f acc=%.4f macro_f1=%.4f",
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_macro_f1"],
            )
        except KeyboardInterrupt:
            logger.warning("Second interrupt received during finalization; partial artifacts remain in %s", run_dir)
        logger.info("Saved outputs to %s", run_dir)


if __name__ == "__main__":
    main()
