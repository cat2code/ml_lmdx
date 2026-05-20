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

from mldmx.datasets.ecal_tpad_dataset import ECalTriggerPadTensorDataset
from mldmx.datasets.ecal_tpad_loading import canonical_axis_from_target_mode, load_ecal_tpad_tensor_events
from mldmx.datasets.preprocess import normalize_continuous_features
from mldmx.datasets.stats import count_classes, target_order_counts
from mldmx.eval.ecal_tpad_slot_model import evaluate
from mldmx.io.artifacts import save_config, save_history, save_json
from mldmx.io.root_files import find_root_files
from mldmx.models import ECalTpadSlotModel
from mldmx.train.checkpoints import load_checkpoint, save_checkpoint
from mldmx.train.ecal_tpad_slot_model import train_one_epoch
from mldmx.train.ecal_tpad_slot_model import ecal_mask_from_event
from mldmx.train.logging import setup_logging
from mldmx.train.modeling import count_trainable_parameters
from mldmx.train.paths import resolve_run_dir
from mldmx.train.progress import make_progress
from mldmx.train.splits import deterministic_split
from mldmx.viz.event_level import plot_event_count_confusion_matrix
from mldmx.viz.ecal import plot_ecal_hit_classes_3d
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
    parser.add_argument("--keep-noise", action="store_true")
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


def resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    for root in (PROJECT_ROOT, PROJECT_ROOT.parent, Path.cwd()):
        candidate = root / path
        if candidate.exists():
            return candidate
    return path


def has_processed_events(processed_dir: Path) -> bool:
    return processed_dir.exists() and any(processed_dir.glob("event_*.pt"))


def load_processed_events(args, processed_dir: Path, logger):
    dataset = ECalTriggerPadTensorDataset(processed_dir)
    limit = len(dataset) if args.max_events is None else min(args.max_events, len(dataset))
    events = []
    event_sources = []
    for dataset_idx in range(limit):
        event = dict(dataset[dataset_idx])
        event_file = dataset.event_files[dataset_idx]
        if "event_idx" not in event:
            event["event_idx"] = torch.tensor(dataset_idx, dtype=torch.long)
        event["source_file"] = event_file.name
        events.append(event)
        event_sources.append(
            {
                "event_idx": dataset_idx,
                "file": event_file.name,
                "entry": dataset_idx,
                "source": "processed",
            }
        )
    logger.info("Loaded %s processed tensor events from %s", len(events), processed_dir)
    return events, event_sources, processed_dir, []


def attach_root_source_metadata(event, source, global_event_idx: int, electron_count: int, source_label: str):
    event["event_idx"] = torch.tensor(global_event_idx, dtype=torch.long)
    event["electron_count"] = torch.tensor(electron_count, dtype=torch.long)
    event["source_label"] = source_label
    event["source_file"] = source["file"]
    event["source_entry"] = int(source["entry"])


def apply_variable_slot_target_mode(event: dict, valid_labels, target_mode: str, max_electrons: int):
    axis = canonical_axis_from_target_mode(target_mode)
    if axis is None:
        event["target_class_names"] = [f"origin {label}" for label in valid_labels]
        event["target_label_order"] = list(valid_labels)
        return event

    if "physical_y" not in event or "ecal_pos" not in event:
        raise KeyError("Canonical slot target mode requires event['physical_y'] and event['ecal_pos'].")

    original_physical_y = event.get("origin_id_y", event["physical_y"]).clone()
    pos = event["ecal_pos"]
    present_labels = sorted({int(label) for label in original_physical_y.tolist()})
    if len(present_labels) > max_electrons:
        raise ValueError(
            f"Event has {len(present_labels)} origin labels, but max_electrons={max_electrons}."
        )

    label_means = []
    for label in present_labels:
        mask = original_physical_y == label
        label_means.append((label, float(pos[mask, axis].mean().item())))
    ordered_labels = [label for label, _mean in sorted(label_means, key=lambda item: (item[1], item[0]))]
    label_to_slot = {label: slot_idx + 1 for slot_idx, label in enumerate(ordered_labels)}

    event["origin_id_y"] = original_physical_y
    event["y"] = torch.tensor(
        [label_to_slot[int(label)] - 1 for label in original_physical_y.tolist()],
        dtype=torch.long,
    )
    event["physical_y"] = event["y"] + 1
    event["target_label_order"] = ordered_labels

    if "fraction_target" in event:
        original_fraction = event.get("origin_id_fraction_target", event["fraction_target"]).clone()
        event["origin_id_fraction_target"] = original_fraction
        if original_fraction.shape[1] == max_electrons + 1:
            original_fraction = original_fraction[:, 1:]
        if original_fraction.shape[1] != len(valid_labels):
            raise ValueError(
                f"Expected {len(valid_labels)} physical-origin fraction columns before canonicalization, "
                f"got {original_fraction.shape[1]}."
            )
        physical_label_to_column = {label: idx for idx, label in enumerate(valid_labels)}
        canonical_fraction = torch.zeros(
            (original_fraction.shape[0], max_electrons),
            dtype=original_fraction.dtype,
            device=original_fraction.device,
        )
        for slot_idx, label in enumerate(ordered_labels):
            source_col = physical_label_to_column.get(int(label))
            if source_col is not None:
                canonical_fraction[:, slot_idx] = original_fraction[:, source_col]
        event["fraction_target"] = canonical_fraction

    axis_name = {0: "x", 1: "y", 2: "z"}[axis]
    event["target_class_names"] = [
        f"slot {slot_idx + 1}: {axis_name}-rank {slot_idx + 1}"
        for slot_idx in range(max_electrons)
    ]
    return event


def apply_target_mode_to_events(events, args):
    for event in events:
        apply_variable_slot_target_mode(
            event,
            valid_labels=tuple(args.valid_labels),
            target_mode=args.target_mode,
            max_electrons=args.max_electrons,
        )


def load_balanced_root_events(args, data_root: Path, logger):
    root_specs = [
        (2, "2e", data_root / "2e/events"),
        (3, "3e", data_root / "3e/events"),
    ]
    filter_noise = not args.keep_noise
    read_step_size = args.read_step_size if args.read_step_size > 0 else None
    events = []
    event_sources = []
    root_files_used = []

    for electron_count, source_label, data_dir in root_specs:
        if not data_dir.exists():
            raise FileNotFoundError(f"Could not find {source_label} ROOT directory: {data_dir}")
        root_files = find_root_files(data_dir)
        root_files_used.extend(root_files)
        logger.info(
            "Loading up to %s %s events from %s",
            args.events_per_class,
            source_label,
            data_dir,
        )
        loaded_events, sources = load_ecal_tpad_tensor_events(
            root_files=root_files,
            max_events=args.events_per_class,
            valid_labels=tuple(args.valid_labels),
            target_mode="physical-origin",
            filter_noise=filter_noise,
            allow_fewer_events=args.allow_fewer_events,
            data_dir=data_dir,
            logger=logger,
            progress_factory=make_progress,
            disable_progress=args.no_progress,
            event_log_every=args.event_log_every,
            read_step_size=read_step_size,
        )
        for event, source in zip(loaded_events, sources):
            global_event_idx = len(events)
            attach_root_source_metadata(
                event,
                source,
                global_event_idx=global_event_idx,
                electron_count=electron_count,
                source_label=source_label,
            )
            source = {
                **source,
                "event_idx": global_event_idx,
                "electron_count": electron_count,
                "source_label": source_label,
                "source_dir": str(data_dir),
            }
            events.append(event)
            event_sources.append(source)

    return events, event_sources, data_root, root_files_used


def load_events(args, logger):
    processed_dir = resolve_path(args.processed_dir)
    if has_processed_events(processed_dir):
        logger.info("Using processed tensor dataset: %s", processed_dir)
        return load_processed_events(args, processed_dir, logger)

    data_root = resolve_path(args.data_root)
    logger.info(
        "Processed dir %s not found or empty; loading balanced ROOT events from %s",
        processed_dir,
        data_root,
    )
    return load_balanced_root_events(args, data_root, logger)


def resolve_requested_device(requested_device: str, logger):
    if requested_device == "cuda":
        if torch.cuda.is_available():
            logger.info("CUDA GPU: %s", torch.cuda.get_device_name(0))
            return torch.device("cuda")
        logger.warning("CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    if requested_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


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
    valid_labels = tuple(args.valid_labels)
    if len(valid_labels) < 2:
        raise ValueError(f"--valid-labels must contain at least two labels, got {valid_labels}.")
    if len(set(valid_labels)) != len(valid_labels):
        raise ValueError(f"--valid-labels contains duplicates: {valid_labels}.")


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
    for idx in train_indices:
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

        plot_ecal_hit_classes_3d(
            pos,
            true_labels,
            run_dir / f"test_ecal_event_{event_idx:04d}_truth.png",
            f"Test event {event_idx} ECal hits, true origin class",
            labels=labels,
        )
        plot_ecal_hit_classes_3d(
            pos,
            pred_labels,
            run_dir / f"test_ecal_event_{event_idx:04d}_predicted.png",
            f"Test event {event_idx} ECal hits, predicted origin class",
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
    device = resolve_requested_device(args.device, logger)
    logger.info("Using device: %s", device)
    logger.info("Noise handling: %s", "keeping noise hits" if args.keep_noise else "filtering noise hits")

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
    apply_target_mode_to_events(events, args)
    logger.info("Training class counts: %s", count_classes(events, splits["train"]))
    logger.info("Validation class counts: %s", count_classes(events, splits["val"]))
    logger.info("Test class counts: %s", count_classes(events, splits["test"]))
    add_class_weights_from_training_split(args, events, splits["train"], logger)

    feature_norm = None
    if not args.no_normalize_features:
        feature_norm = normalize_continuous_features(events, splits["train"])
        logger.info("Normalized continuous feature columns from training split statistics.")

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
