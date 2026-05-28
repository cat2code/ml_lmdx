"""
Train the ECal/TPAD slot-validity multi-task model.

Example from the repository root:

    python mldmx/scripts/train_ecal_tpad_slot_model.py --events-per-class 10 --epochs 2 --device cpu

Example from the mldmx directory:

    pip install -e .; python scripts/train_ecal_tpad_slot_model.py --events-per-class 10 --epochs 2
"""

import argparse
from collections import Counter
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    apply_variable_count_target_mode_to_events,
    filter_noise_tensor_event,
    load_grouped_root_tensor_events,
    load_multi_sharded_tensor_events,
    load_or_create_sharded_tensor_events,
    load_processed_or_grouped_root_tensor_events,
)
from mldmx.datasets.ecal_tpad_shards import MultiShardedECalTpadDataset, ShardedECalTpadDataset
from mldmx.datasets.preprocess import (
    fit_continuous_feature_normalization,
    normalize_continuous_features,
    normalize_event_continuous_features,
)
from mldmx.datasets.stats import count_classes, target_order_counts
from mldmx.eval.ecal_tpad_slot_model import evaluate
from mldmx.io.artifacts import save_config, save_history, save_json
from mldmx.models import ECalTpadSlotModel
from mldmx.train.checkpoints import load_checkpoint, save_checkpoint
from mldmx.train.ecal_tpad_slot_model import train_one_epoch
from mldmx.train.ecal_tpad_slot_model import ecal_mask_from_event
from mldmx.train.logging import setup_logging
from mldmx.train.modeling import count_trainable_parameters
from mldmx.train.paths import resolve_existing_path, resolve_run_dir
from mldmx.train.progress import make_progress
from mldmx.train.splits import deterministic_split
from mldmx.train.utils import resolve_device
from mldmx.viz.event_level import plot_event_count_confusion_matrix
from mldmx.viz.ecal import plot_ecal_truth_prediction_pair
from mldmx.viz.training import plot_confusion_matrix, plot_history


VALID_LABELS = (1, 2, 3)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/ldmx_overlay_events_700k"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ecal_tpad_slot_model"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/ecal_tpad_slot_model"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ECalTpadSlotModel on balanced 2e/3e ECal + TriggerPadTracks events."
    )
    parser.add_argument("--events-per-class", type=int, default=10)
    parser.add_argument("--max-events", type=int, default=None, help="Optional limit for processed datasets.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--processed-cache",
        type=Path,
        default=None,
        help="ML-ready sharded cache to reuse or create from --data-root (one ROOT file per shard).",
    )
    parser.add_argument(
        "--processed-cache-root",
        type=Path,
        default=None,
        help="Directory containing separate 2e/events and 3e/events sharded caches.",
    )
    parser.add_argument(
        "--processed-source",
        action="append",
        nargs=3,
        metavar=("ELECTRON_COUNT", "LABEL", "CACHE_DIR"),
        help="Add one existing sharded cache source, e.g. --processed-source 2 2e data/.../2e/events.",
    )
    parser.add_argument(
        "--events-per-source",
        type=int,
        default=None,
        help="Balanced limit per processed source for multi-source sharded caches; total is sources times this value.",
    )
    parser.add_argument("--force-sharded-cache", action="store_true", help="Rebuild requested sharded cache files.")
    parser.add_argument(
        "--allow-incomplete-sharded-cache",
        action="store_true",
        help="Train only from already completed valid shards instead of completing a partial cache.",
    )
    parser.add_argument("--max-cache-root-files", type=int, default=None, help="Limit ROOT files per source when creating a smoke cache.")
    parser.add_argument("--max-events-per-root-file", type=int, default=None, help="Limit events per ROOT shard when creating a smoke cache.")
    parser.add_argument("--shard-cache-size", type=int, default=1, help="Number of recently loaded processed shards retained in RAM.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory name under --output-root. Defaults to a timestamped run name.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8, help="Number of events per optimizer step.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Defaults to CUDA when available, otherwise CPU.",
    )
    parser.add_argument("--valid-labels", type=int, nargs="+", default=list(VALID_LABELS))
    parser.add_argument(
        "--target-mode",
        choices=("canonical-y", "canonical-x", "canonical-z", "physical-origin"),
        default="canonical-y",
        help=(
            "canonical-* maps the physical origin IDs present in each event into variable-count "
            "slots ordered by the requested coordinate axis."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--max-electrons", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-type-embedding", action="store_true")
    parser.add_argument("--lambda-origin", type=float, default=1.0)
    parser.add_argument("--lambda-fraction", type=float, default=1.0)
    parser.add_argument("--lambda-slot", type=float, default=0.5)
    parser.add_argument("--lambda-count", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", choices=("none", "plateau"), default="none")
    parser.add_argument("--plateau-patience", type=int, default=3)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency weights for hit-origin and event-count CE losses.",
    )
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="Unsupported for slot training; use --supervise-noise for explicit background labels.",
    )
    parser.add_argument(
        "--supervise-noise",
        action="store_true",
        help=(
            "Advanced model only: retain stored/ROOT noise_flag hits and assign them "
            "to background class 0 with background-only fraction targets."
        ),
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--num-ecal-plots",
        type=int,
        default=2,
        help="Number of test events to save as truth/predicted ECal 3D origin plots.",
    )
    parser.add_argument("--event-log-every", type=int, default=0)
    parser.add_argument(
        "--read-step-size",
        type=int,
        default=500,
        help="Number of ROOT entries per chunk while reading. Use 0 for one read per file.",
    )
    parser.add_argument(
        "--allow-fewer-events",
        action="store_true",
        help="Continue if fewer ROOT events than --events-per-class are found for a class.",
    )
    args = parser.parse_args()
    args.output_dir = args.output_root
    return args


def load_events(args, logger):
    data_root = resolve_existing_path(args.data_root, project_root=PROJECT_ROOT)
    root_specs = [
        (2, "2e", data_root / "2e/events"),
        (3, "3e", data_root / "3e/events"),
    ]
    read_step_size = args.read_step_size if args.read_step_size > 0 else None
    if args.processed_cache_root is not None or args.processed_source is not None:
        processed_sources = processed_sources_from_args(args)
        logger.info(
            "Multi-source sharded cache mode selected; use --events-per-source for balanced staged runs."
        )
        return load_multi_sharded_tensor_events(
            processed_sources=processed_sources,
            max_events=args.max_events,
            events_per_source=args.events_per_source,
            shard_cache_size=args.shard_cache_size,
            allow_incomplete_cache=args.allow_incomplete_sharded_cache,
            logger=logger,
        )
    if args.processed_cache is not None:
        processed_cache = args.processed_cache
        if not processed_cache.is_absolute():
            processed_cache = PROJECT_ROOT / processed_cache
        logger.info(
            "Sharded cache mode selected; --events-per-class applies only to the legacy ROOT path. "
            "The cache represents its selected ROOT files."
        )
        return load_or_create_sharded_tensor_events(
            processed_cache=processed_cache,
            root_specs=root_specs,
            valid_labels=tuple(args.valid_labels),
            max_events=args.max_events,
            filter_noise=False,
            supervise_noise=True,
            force=args.force_sharded_cache,
            max_root_files=args.max_cache_root_files,
            max_events_per_root_file=args.max_events_per_root_file,
            shard_cache_size=args.shard_cache_size,
            allow_incomplete_cache=args.allow_incomplete_sharded_cache,
            logger=logger,
            read_step_size=read_step_size,
        )
    processed_dir = resolve_existing_path(args.processed_dir, project_root=PROJECT_ROOT)
    if args.supervise_noise:
        logger.info("Explicit noise supervision selected; bypassing processed tensors and reading ROOT inputs.")
        events, event_sources, root_files = load_grouped_root_tensor_events(
            root_specs=root_specs,
            events_per_source=args.events_per_class,
            valid_labels=tuple(args.valid_labels),
            filter_noise=False,
            supervise_noise=True,
            allow_fewer_events=args.allow_fewer_events,
            logger=logger,
            progress_factory=make_progress,
            disable_progress=args.no_progress,
            event_log_every=args.event_log_every,
            read_step_size=read_step_size,
        )
        return events, event_sources, data_root, root_files
    return load_processed_or_grouped_root_tensor_events(
        processed_dir=processed_dir,
        root_specs=root_specs,
        root_data_dir=data_root,
        events_per_source=args.events_per_class,
        max_processed_events=args.max_events,
        valid_labels=tuple(args.valid_labels),
        filter_noise=not args.keep_noise,
        supervise_noise=False,
        allow_fewer_events=args.allow_fewer_events,
        logger=logger,
        progress_factory=make_progress,
        disable_progress=args.no_progress,
        event_log_every=args.event_log_every,
        read_step_size=read_step_size,
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


def model_kwargs_from_args(args, input_dim: int):
    return {
        "in_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "max_electrons": args.max_electrons,
        "dropout": args.dropout,
        "use_type_embedding": not args.no_type_embedding,
    }


def validate_args(args):
    if args.events_per_class <= 0:
        raise ValueError("--events-per-class must be positive.")
    if args.max_events is not None and args.max_events <= 0:
        raise ValueError("--max-events must be positive when provided.")
    for name in ("max_cache_root_files", "max_events_per_root_file", "shard_cache_size", "events_per_source"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided.")
    selected_processed_modes = sum(
        value is not None for value in (args.processed_cache, args.processed_cache_root, args.processed_source)
    )
    if selected_processed_modes > 1:
        raise ValueError("Use only one of --processed-cache, --processed-cache-root, or --processed-source.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive.")
    if args.read_step_size < 0:
        raise ValueError("--read-step-size must be non-negative.")
    if args.max_electrons < 3:
        raise ValueError("--max-electrons must be at least 3 for current 2e/3e data.")
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError("--hidden-dim must be divisible by --num-heads.")
    if args.keep_noise:
        raise ValueError(
            "--keep-noise does not define slot-model background labels; "
            "use --supervise-noise for the explicit advanced-model experiment."
        )
    if args.supervise_noise and args.target_mode != "canonical-y":
        raise ValueError("--supervise-noise currently requires --target-mode canonical-y.")
    valid_labels = tuple(args.valid_labels)
    if len(valid_labels) < 2:
        raise ValueError(f"--valid-labels must contain at least two labels, got {valid_labels}.")
    if len(set(valid_labels)) != len(valid_labels):
        raise ValueError(f"--valid-labels contains duplicates: {valid_labels}.")


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return PROJECT_ROOT / path


def processed_sources_from_args(args):
    if args.processed_cache_root is not None:
        root = resolve_project_path(args.processed_cache_root)
        return [
            (2, "2e", root / "2e/events"),
            (3, "3e", root / "3e/events"),
        ]
    return [
        (int(electron_count), label, resolve_project_path(cache_dir))
        for electron_count, label, cache_dir in (args.processed_source or [])
    ]


def prepare_targets_and_features(events, splits, args, logger):
    if isinstance(events, (ShardedECalTpadDataset, MultiShardedECalTpadDataset)):
        def target_transform(event):
            if not args.supervise_noise:
                event = filter_noise_tensor_event(event)
            return apply_variable_count_target_mode(
                event,
                valid_labels=tuple(args.valid_labels),
                target_mode=args.target_mode,
                max_electrons=args.max_electrons,
            )

        events.set_event_transform(target_transform)
        feature_norm = None
        if not args.no_normalize_features:
            feature_norm = fit_continuous_feature_normalization(events, splits["train"])

            def target_and_feature_transform(event):
                return normalize_event_continuous_features(target_transform(event), feature_norm)

            events.set_event_transform(target_and_feature_transform)
            logger.info("Fitted lazy canonical-event feature normalization from sharded training events.")
        return feature_norm

    apply_variable_count_target_mode_to_events(
        events,
        valid_labels=tuple(args.valid_labels),
        target_mode=args.target_mode,
        max_electrons=args.max_electrons,
    )
    if not args.no_normalize_features:
        feature_norm = normalize_continuous_features(events, splits["train"])
        logger.info("Normalized continuous feature columns from training split statistics.")
        return feature_norm
    return None


def inverse_frequency_weights(counts: dict[int, int], num_classes: int):
    weights = [0.0] * num_classes
    positive = {idx: count for idx, count in counts.items() if count > 0}
    if not positive:
        return [1.0] * num_classes
    total = sum(positive.values())
    for idx, count in positive.items():
        if 0 <= idx < num_classes:
            weights[idx] = total / (len(positive) * count)
    nonzero = [value for value in weights if value > 0.0]
    mean_nonzero = sum(nonzero) / len(nonzero)
    return [value / mean_nonzero if value > 0.0 else 0.0 for value in weights]


def add_class_weights_from_training_split(args, events, train_indices, logger):
    if args.no_class_weights:
        args.origin_class_weights = None
        args.count_class_weights = None
        return

    origin_counts = Counter()
    count_counts = Counter()
    ordered_indices = (
        events.order_indices_for_access(train_indices)
        if hasattr(events, "order_indices_for_access")
        else train_indices
    )
    for idx in ordered_indices:
        event = events[idx]
        origin_counts.update(int(label) for label in event["physical_y"].tolist())
        if "electron_count" in event:
            value = event["electron_count"]
            if isinstance(value, torch.Tensor):
                value = int(value.detach().cpu().reshape(-1)[0].item())
            else:
                value = int(value)
        else:
            value = len({int(label) for label in event["physical_y"].tolist()})
        count_counts.update([value])

    args.origin_class_weights = inverse_frequency_weights(
        origin_counts,
        num_classes=args.max_electrons + 1,
    )
    args.count_class_weights = inverse_frequency_weights(
        count_counts,
        num_classes=args.max_electrons + 1,
    )
    logger.info("Origin CE class weights: %s", args.origin_class_weights)
    logger.info("Count CE class weights: %s", args.count_class_weights)


def save_event_count_plots(run_dir, predictions, args, stem="event_count_confusion_matrix"):
    if not predictions:
        return
    y_true = [row["true_count"] for row in predictions]
    y_pred = [row["predicted_count"] for row in predictions]
    labels = list(range(args.max_electrons + 1))
    fig, _ax = plot_event_count_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        title="Event electron count confusion matrix",
        output_path=run_dir / f"{stem}.png",
        normalize=False,
    )
    plt.close(fig)

    fig, _ax = plot_event_count_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        title="Event electron count confusion matrix, normalized",
        output_path=run_dir / f"{stem}_normalized.png",
        normalize=True,
    )
    plt.close(fig)


@torch.no_grad()
def plot_test_ecal_hit_classes(model, events, test_indices, args, device, run_dir):
    if not test_indices or args.num_ecal_plots <= 0:
        return

    model.eval()
    labels = list(range(args.max_electrons + 1))
    for event_idx in test_indices[: args.num_ecal_plots]:
        event = events[event_idx]
        x = event["x"].to(device=device, dtype=torch.float32)
        ecal_mask = ecal_mask_from_event(event).to(device)
        outputs = model(x, ecal_mask=ecal_mask)
        pred_labels = outputs["origin_logits"][ecal_mask].argmax(dim=1).detach().cpu()

        if "physical_y" in event:
            true_labels = event["physical_y"].detach().cpu()
        else:
            true_labels = event["y"].detach().cpu() + 1
        pos = event["ecal_pos"].detach().cpu()

        plot_ecal_truth_prediction_pair(
            pos,
            true_labels,
            pred_labels,
            truth_path=run_dir / f"test_ecal_event_{event_idx:04d}_truth.png",
            predicted_path=run_dir / f"test_ecal_event_{event_idx:04d}_predicted.png",
            truth_title=f"Test event {event_idx} ECal hits, true origin class",
            predicted_title=f"Test event {event_idx} ECal hits, predicted origin class",
            labels=labels,
        )


def main():
    args = parse_args()
    validate_args(args)
    run_dir = resolve_run_dir(args)
    logger = setup_logging(run_dir)
    torch.manual_seed(args.seed)

    logger.info("Output directory: %s", run_dir)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    device = resolve_device(args.device, logger)
    logger.info("Using device: %s", device)
    if args.supervise_noise:
        logger.info("Noise handling: retaining stored noise hits with explicit background-class supervision")
    elif args.keep_noise:
        logger.info("Noise handling: retaining noise hits without explicit background-class supervision")
    else:
        logger.info("Noise handling: filtering noise hits at training access time; background class receives no hit supervision")

    events, event_sources, data_dir, root_files = load_events(args, logger)
    if len(events) < 20:
        raise ValueError(
            f"Need at least 20 events for the existing 80/15/5 split; loaded {len(events)}."
        )
    splits = deterministic_split(len(events), args.seed)
    logger.info(
        "Split sizes: train=%s val=%s test=%s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    logger.info("Target mode: %s", args.target_mode)
    feature_norm = prepare_targets_and_features(events, splits, args, logger)
    if args.supervise_noise:
        num_noise = sum(int(event["is_noise_target"].sum().item()) for event in events)
        logger.info("Explicit background targets retained: %s noise ECal hits", num_noise)
    logger.info("Training class counts: %s", count_classes(events, splits["train"]))
    logger.info("Validation class counts: %s", count_classes(events, splits["val"]))
    logger.info("Test class counts: %s", count_classes(events, splits["test"]))
    add_class_weights_from_training_split(args, events, splits["train"], logger)

    input_dim = int(events[0]["x"].shape[1])
    model_kwargs = model_kwargs_from_args(args, input_dim=input_dim)
    model = ECalTpadSlotModel(**model_kwargs).to(device)
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
                "Resume with the same data selection and --seed used for the checkpoint."
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
            train_metrics = train_one_epoch(
                model,
                events,
                splits["train"],
                optimizer,
                args,
                device,
                epoch,
                logger,
            )
            val_start = time.time()
            val_metrics, _val_predictions = evaluate(model, events, splits["val"], args, device, "val")
            val_metrics["val_elapsed_sec"] = time.time() - val_start
            epoch_metrics.update(train_metrics)
            epoch_metrics.update(val_metrics)
            history.append(epoch_metrics)

            logger.info(
                "epoch=%03d val_loss=%.5f val_hit_acc=%.4f val_count_acc=%.4f lr=%.3g",
                epoch + 1,
                val_metrics["val_loss"],
                val_metrics["val_accuracy"],
                val_metrics["val_count_accuracy"],
                lr,
            )

            if scheduler is not None:
                scheduler.step(val_metrics["val_loss"])

            save_history(history, run_dir)
            plot_history(history, run_dir)

            save_checkpoint(
                run_dir / "checkpoints/latest.pt",
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
            test_metrics, test_predictions = evaluate(
                model,
                events,
                splits["test"],
                args,
                device,
                "test",
                collect_predictions=True,
            )
            test_metrics["test_elapsed_sec"] = time.time() - test_start
            val_metrics, val_predictions = evaluate(
                model,
                events,
                splits["val"],
                args,
                device,
                "final_val",
                collect_predictions=True,
            )
            final_metrics = {
                "interrupted": interrupted,
                "best_val_loss": best_val_loss,
                **val_metrics,
                **test_metrics,
            }
            save_json(run_dir / "final_metrics.json", final_metrics)
            save_json(run_dir / "test_event_predictions.json", test_predictions)
            save_json(run_dir / "val_event_predictions.json", val_predictions)

            plot_confusion_matrix(
                test_metrics["test_hit_confusion"],
                list(range(args.max_electrons + 1)),
                run_dir / "test_hit_origin_confusion_matrix.png",
                "Test hit-origin confusion matrix",
            )
            save_event_count_plots(
                run_dir,
                test_predictions,
                args,
                stem="event_count_confusion_matrix",
            )
            save_event_count_plots(
                run_dir,
                val_predictions,
                args,
                stem="val_event_count_confusion_matrix",
            )
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
                "Final test: loss=%.5f hit_acc=%.4f count_acc=%.4f",
                test_metrics["test_loss"],
                test_metrics["test_accuracy"],
                test_metrics["test_count_accuracy"],
            )
        except KeyboardInterrupt:
            logger.warning("Second interrupt received during finalization; partial artifacts remain in %s", run_dir)
        logger.info("Saved outputs to %s", run_dir)


if __name__ == "__main__":
    main()
