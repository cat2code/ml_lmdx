"""
Scaled ECAL/TPAD MLPF-lite training script.

Example from the repository root:

    python mldmx/scripts/train_ecal_tpad_mlpf_lite_scaled.py --max-events 1000 --epochs 1

Example from the mldmx directory:

    pip install -e .; python scripts/train_ecal_tpad_mlpf_lite_scaled.py --max-events 15000 --epochs 20

    pip install -e .; python scripts/train_ecal_tpad_mlpf_lite_scaled.py --max-events 4000 --epochs 20
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_dataset import (
    ECalTriggerPadTensorDataset,
    save_tensor_event,
    write_manifest,
)
from mldmx.datasets.ecal_tpad_loading import load_ecal_tpad_tensor_events
from mldmx.datasets.preprocess import normalize_continuous_features
from mldmx.datasets.stats import count_classes, target_order_counts
from mldmx.eval.ecal_tpad_mlpf_lite import evaluate
from mldmx.io.artifacts import save_config, save_history, save_json
from mldmx.io.root_files import find_root_files
from mldmx.models import ECalTpadMLPFLiteTransformer
from mldmx.train.checkpoints import load_checkpoint, save_checkpoint
from mldmx.train.ecal_tpad_mlpf_lite import train_one_epoch
from mldmx.train.logging import setup_logging
from mldmx.train.modeling import count_trainable_parameters, model_kwargs_from_args
from mldmx.train.paths import resolve_data_dir, resolve_run_dir
from mldmx.train.progress import make_progress
from mldmx.train.splits import deterministic_split
from mldmx.train.utils import resolve_device
from mldmx.viz.ecal import plot_ecal_hit_classes_3d
from mldmx.viz.training import plot_confusion_matrix, plot_history, plot_test_fraction_summaries


VALID_LABELS = (1, 2, 3)
DEFAULT_DATA_DIR = Path("data/ldmx_overlay_events_700k/3e/events")
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ecal_tpad_mlpf_lite_scaled"
CACHE_SCHEMA_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the ECAL/TPAD MLPF-lite transformer on a configurable ROOT event subset."
    )
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=(
            "Directory used for cached tensorized events. A config-specific subdirectory "
            "is created and reused when its manifest matches the requested ROOT inputs."
        ),
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Rebuild the tensor cache even if a matching manifest already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/ecal_tpad_mlpf_lite_scaled",
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


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _root_file_metadata(root_files):
    metadata = []
    for path in root_files:
        stat = path.stat()
        metadata.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return metadata


def tensor_cache_spec(args, data_dir, root_files):
    filter_noise = not args.keep_noise
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "reader": "ecal_tpad_mlpf_lite_scaled",
        "data_dir": str(data_dir.resolve()),
        "root_files": _root_file_metadata(root_files),
        "max_events": int(args.max_events),
        "valid_labels": [int(label) for label in args.valid_labels],
        "target_mode": args.target_mode,
        "filter_noise": bool(filter_noise),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = hashlib.sha256(encoded).hexdigest()[:16]
    return payload, signature


def resolve_processed_cache_dir(args, data_dir, root_files):
    processed_root = args.processed_dir
    if not processed_root.is_absolute():
        processed_root = PROJECT_ROOT / processed_root
    spec, signature = tensor_cache_spec(args, data_dir, root_files)
    return processed_root / f"cache_{signature}", spec, signature


def _manifest_matches_cache(manifest, spec, signature):
    return (
        manifest.get("cache_signature") == signature
        and manifest.get("cache_spec") == spec
        and bool(manifest.get("event_files"))
    )


def has_matching_tensor_cache(cache_dir, spec, signature):
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if not _manifest_matches_cache(manifest, spec, signature):
        return False
    return all((cache_dir / name).exists() for name in manifest["event_files"])


def load_cached_tensor_events(cache_dir, logger):
    dataset = ECalTriggerPadTensorDataset(cache_dir)
    events = []
    for dataset_idx in range(len(dataset)):
        event = dict(dataset[dataset_idx])
        if "event_idx" not in event:
            event["event_idx"] = torch.tensor(dataset_idx, dtype=torch.long)
        events.append(event)

    event_sources = dataset.metadata.get("event_sources", [])
    if not event_sources:
        event_sources = [
            {
                "event_idx": idx,
                "file": dataset.event_files[idx].name,
                "entry": idx,
                "source": "processed",
            }
            for idx in range(len(dataset))
        ]
    logger.info("Loaded %s tensorized events from cache: %s", len(events), cache_dir)
    return events, event_sources


def write_tensor_cache(cache_dir, events, event_sources, spec, signature, elapsed_sec, logger):
    cache_dir.mkdir(parents=True, exist_ok=True)
    event_files = []
    for event_idx, event in enumerate(events):
        event_files.append(save_tensor_event(event, cache_dir, event_idx))

    metadata = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_signature": signature,
        "cache_spec": spec,
        "num_events": len(event_files),
        "event_sources": _json_ready(event_sources),
        "preprocess_elapsed_sec": float(elapsed_sec),
        "feature_layout": [
            "is_ecal",
            "is_tpad",
            "ecal_x",
            "ecal_y",
            "ecal_z",
            "ecal_energy",
            "tpad_centroid",
            "tpad_pe",
        ],
    }
    manifest_path = write_manifest(cache_dir, metadata=metadata, event_files=event_files)
    logger.info("Wrote tensor cache manifest: %s", manifest_path)


def load_tensor_events(args, data_dir, root_files, logger):
    cache_dir, spec, signature = resolve_processed_cache_dir(args, data_dir, root_files)
    if not args.force_preprocess and has_matching_tensor_cache(cache_dir, spec, signature):
        logger.info("Using matching tensor cache signature %s", signature)
        return load_cached_tensor_events(cache_dir, logger), cache_dir

    filter_noise = not args.keep_noise
    read_step_size = args.read_step_size if args.read_step_size > 0 else None
    if args.force_preprocess:
        logger.info("Force preprocessing requested; rebuilding tensor cache in %s", cache_dir)
    else:
        logger.info("No matching tensor cache found; preprocessing ROOT events into %s", cache_dir)
    preprocess_start = time.perf_counter()
    events, event_sources = load_ecal_tpad_tensor_events(
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
    preprocess_elapsed = time.perf_counter() - preprocess_start
    logger.info(
        "Preprocessed %s events in %.2f seconds (%.3f sec/event)",
        len(events),
        preprocess_elapsed,
        preprocess_elapsed / max(len(events), 1),
    )
    write_tensor_cache(
        cache_dir,
        events,
        event_sources,
        spec=spec,
        signature=signature,
        elapsed_sec=preprocess_elapsed,
        logger=logger,
    )
    return (events, event_sources), cache_dir


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

    data_dir = resolve_data_dir(args.data_dir, project_root=PROJECT_ROOT)
    root_files = find_root_files(data_dir)
    logger.info("Output directory: %s", run_dir)
    logger.info("Data directory: %s", data_dir)
    logger.info("First ROOT files: %s", ", ".join(path.name for path in root_files[:5]))
    logger.info("Noise handling: %s", "keeping noise hits" if args.keep_noise else "filtering noise hits")

    device = resolve_device(args.device, logger)
    logger.info("Using device: %s", device)

    (events, event_sources), processed_cache_dir = load_tensor_events(args, data_dir, root_files, logger)
    args.resolved_processed_cache_dir = processed_cache_dir
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
