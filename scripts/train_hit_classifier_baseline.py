"""Train one maintained ECal hit-origin classification baseline."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    apply_variable_count_target_mode_to_events,
    filter_noise_tensor_event,
    load_or_create_sharded_tensor_events,
    load_multi_sharded_tensor_events,
    load_processed_or_grouped_root_tensor_events,
)
from mldmx.datasets.ecal_tpad_shards import MultiShardedECalTpadDataset, ShardedECalTpadDataset
from mldmx.datasets.model_views import (
    ecal_gravnet_view,
    ecal_tpad_gravnet_view,
    ecal_tpad_transformer_view,
    ecal_transformer_view,
)
from mldmx.datasets.preprocess import (
    fit_continuous_feature_normalization,
    normalize_continuous_features,
    normalize_event_continuous_features,
)
from mldmx.datasets.stats import count_classes, target_order_counts
from mldmx.eval.hit_classifier_baseline import evaluate
from mldmx.io.artifacts import save_config, save_history, save_json
from mldmx.models import ECalGravNet, ECalTpadGravNet, ECalTpadTransformer, ECalTransformer
from mldmx.train.checkpoints import load_checkpoint, save_checkpoint
from mldmx.train.hit_classifier_baseline import compute_event_losses, train_one_epoch
from mldmx.train.logging import setup_logging
from mldmx.train.modeling import count_trainable_parameters
from mldmx.train.paths import resolve_existing_path, resolve_run_dir
from mldmx.train.progress import make_progress
from mldmx.train.splits import deterministic_split
from mldmx.train.utils import resolve_device
from mldmx.viz.ecal import plot_ecal_truth_prediction_pair
from mldmx.viz.training import plot_confusion_matrix, plot_history


MODEL_NAMES = ("ECalGravNet", "ECalTpadGravNet", "ECalTransformer", "ECalTpadTransformer")
VALID_LABELS = (1, 2, 3)
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ecal_tpad_hit_classifier_baseline"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/ldmx_overlay_events_700k"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/hit_classifier_baseline"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a maintained baseline using canonical-y ECal hit-origin classification."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
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
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--valid-labels", type=int, nargs="+", default=list(VALID_LABELS))
    parser.add_argument(
        "--target-mode",
        choices=("canonical-y",),
        default="canonical-y",
        help="Maintained comparisons use canonical ordering along ECal y.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--space-dimensions", type=int, default=4)
    parser.add_argument("--propagate-dimensions", type=int, default=32)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-normalize-features", action="store_true")
    parser.add_argument(
        "--cache-model-views",
        action="store_true",
        help=(
            "Materialize the selected adapter view once for reuse across epochs; "
            "can reduce repeated view work at the cost of extra memory for ECal-only inputs."
        ),
    )
    parser.add_argument(
        "--allow-small-split",
        action="store_true",
        help="Permit non-empty deterministic splits below 20 events for smoke validation.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--num-ecal-plots", type=int, default=2)
    parser.add_argument("--event-log-every", type=int, default=0)
    parser.add_argument("--read-step-size", type=int, default=500)
    parser.add_argument("--allow-fewer-events", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_root
    return args


def validate_args(args):
    if args.events_per_class <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--events-per-class, --epochs, and --batch-size must be positive.")
    if args.max_events is not None and args.max_events <= 0:
        raise ValueError("--max-events must be positive when provided.")
    for name in ("max_cache_root_files", "max_events_per_root_file", "shard_cache_size", "events_per_source"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive when provided.")
    if args.checkpoint_every <= 0 or args.read_step_size < 0:
        raise ValueError("--checkpoint-every must be positive and --read-step-size non-negative.")
    if len(args.valid_labels) < 2 or len(set(args.valid_labels)) != len(args.valid_labels):
        raise ValueError("--valid-labels must contain at least two distinct origin labels.")
    if args.hidden_dim % args.num_heads != 0 and "Transformer" in args.model:
        raise ValueError("--hidden-dim must be divisible by --num-heads for transformer models.")
    selected_processed_modes = sum(
        value is not None for value in (args.processed_cache, args.processed_cache_root, args.processed_source)
    )
    if selected_processed_modes > 1:
        raise ValueError("Use only one of --processed-cache, --processed-cache-root, or --processed-source.")


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
    events, event_sources, data_dir, root_files = load_processed_or_grouped_root_tensor_events(
        processed_dir=processed_dir,
        root_specs=root_specs,
        root_data_dir=data_root,
        events_per_source=args.events_per_class,
        max_processed_events=args.max_events,
        valid_labels=tuple(args.valid_labels),
        filter_noise=True,
        supervise_noise=False,
        allow_fewer_events=args.allow_fewer_events,
        logger=logger,
        progress_factory=make_progress,
        disable_progress=args.no_progress,
        event_log_every=args.event_log_every,
        read_step_size=read_step_size,
    )
    if isinstance(events, (ShardedECalTpadDataset, MultiShardedECalTpadDataset)):
        logger.info("Training lazily from ML-ready processed shards: %s", data_dir)
    elif not root_files:
        manifest_path = Path(data_dir) / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as handle:
                filter_noise = json.load(handle).get("filter_noise")
            if filter_noise is False:
                raise ValueError(
                    "Maintained baseline training requires noise-filtered canonical events; "
                    "the selected processed manifest has filter_noise=false."
                )
            if filter_noise is True:
                logger.info("Processed manifest confirms noise-filtered ECal hits.")
            else:
                logger.warning("Processed manifest does not state filter_noise; noise policy is unverifiable.")
        else:
            logger.warning("Processed input has no manifest; noise policy is unverifiable.")
    else:
        logger.info("ROOT inputs tensorized with noise filtering enabled for baseline training.")
    return events, event_sources, data_dir, root_files


def prepare_targets_and_features(events, splits, args, logger):
    if isinstance(events, (ShardedECalTpadDataset, MultiShardedECalTpadDataset)):
        def target_transform(event):
            event = filter_noise_tensor_event(event)
            return apply_variable_count_target_mode(
                event,
                valid_labels=tuple(args.valid_labels),
                target_mode=args.target_mode,
                max_electrons=len(args.valid_labels),
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
        max_electrons=len(args.valid_labels),
    )
    if not args.no_normalize_features:
        feature_norm = normalize_continuous_features(events, splits["train"])
        logger.info("Normalized canonical combined continuous feature columns from training events.")
        return feature_norm
    return None


def model_and_view(args, input_dim):
    out_dim = len(args.valid_labels)
    if args.model == "ECalGravNet":
        return (
            ECalGravNet(
                in_dim=input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=out_dim,
                num_layers=args.num_layers,
                space_dimensions=args.space_dimensions,
                propagate_dimensions=args.propagate_dimensions,
                k=args.k,
                dropout=args.dropout,
            ),
            ecal_gravnet_view,
        )
    if args.model == "ECalTpadGravNet":
        return (
            ECalTpadGravNet(
                in_dim=input_dim,
                hidden_dim=args.hidden_dim,
                out_dim=out_dim,
                num_layers=args.num_layers,
                space_dimensions=args.space_dimensions,
                propagate_dimensions=args.propagate_dimensions,
                k=args.k,
                dropout=args.dropout,
            ),
            ecal_tpad_gravnet_view,
        )
    transformer_kwargs = {
        "in_dim": input_dim,
        "d_model": args.hidden_dim,
        "nhead": args.num_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "out_dim": out_dim,
    }
    if args.model == "ECalTransformer":
        return ECalTransformer(**transformer_kwargs), ecal_transformer_view
    return ECalTpadTransformer(**transformer_kwargs), ecal_tpad_transformer_view


def model_kwargs_from_args(args, input_dim):
    if "GravNet" in args.model:
        return {
            "in_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "out_dim": len(args.valid_labels),
            "num_layers": args.num_layers,
            "space_dimensions": args.space_dimensions,
            "propagate_dimensions": args.propagate_dimensions,
            "k": args.k,
            "dropout": args.dropout,
        }
    return {
        "in_dim": input_dim,
        "d_model": args.hidden_dim,
        "nhead": args.num_heads,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "out_dim": len(args.valid_labels),
    }


@torch.no_grad()
def plot_test_predictions(model, events_or_views, test_indices, view_fn, args, device, run_dir):
    model.eval()
    labels = list(range(len(args.valid_labels)))
    for event_idx in test_indices[: args.num_ecal_plots]:
        losses = compute_event_losses(model, events_or_views[event_idx], view_fn, device)
        view = losses["view"]
        plot_ecal_truth_prediction_pair(
            view["ecal_pos"].detach().cpu(),
            losses["true_class"].detach().cpu(),
            losses["pred_class"].detach().cpu(),
            truth_path=run_dir / f"test_ecal_event_{event_idx:04d}_truth.png",
            predicted_path=run_dir / f"test_ecal_event_{event_idx:04d}_predicted.png",
            truth_title=f"{args.model} test event {event_idx}, canonical-y truth",
            predicted_title=f"{args.model} test event {event_idx}, canonical-y prediction",
            labels=labels,
        )


def main():
    args = parse_args()
    validate_args(args)
    run_dir = resolve_run_dir(args)
    logger = setup_logging(run_dir, logger_name="hit_classifier_baseline")
    torch.manual_seed(args.seed)

    device = resolve_device(args.device, logger)
    logger.info("Output directory: %s", run_dir)
    logger.info("Model: %s", args.model)
    logger.info("Using device: %s", device)
    logger.info("Target mode: canonical-y; baseline training noise policy: filtered")

    events, event_sources, data_dir, root_files = load_events(args, logger)
    splits = deterministic_split(len(events), args.seed, allow_small=args.allow_small_split)
    logger.info(
        "Split sizes: train=%s val=%s test=%s",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    feature_norm = prepare_targets_and_features(events, splits, args, logger)
    logger.info("Training canonical class counts: %s", count_classes(events, splits["train"]))

    prototype_view = {
        "ECalGravNet": ecal_gravnet_view,
        "ECalTpadGravNet": ecal_tpad_gravnet_view,
        "ECalTransformer": ecal_transformer_view,
        "ECalTpadTransformer": ecal_tpad_transformer_view,
    }[args.model](events[0])
    input_dim = int(prototype_view["x"].shape[1])
    model, view_fn = model_and_view(args, input_dim)
    model = model.to(device)
    model_kwargs = model_kwargs_from_args(args, input_dim)
    training_events = events
    training_view_fn = view_fn
    logger.info("Input feature dimension: %s", input_dim)
    if args.cache_model_views:
        if isinstance(events, (ShardedECalTpadDataset, MultiShardedECalTpadDataset)):
            logger.warning("--cache-model-views materializes selected sharded events in RAM.")
        training_events = [view_fn(event) for event in events]
        training_view_fn = None
        logger.info("Prepared %s model-specific event views once for epoch reuse.", len(training_events))
    else:
        logger.info("Deriving model-specific views on demand; pass --cache-model-views to reuse them.")
    logger.info("Trainable parameters: %s", count_trainable_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    history = []
    best_val_loss = float("inf")
    start_epoch = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        if checkpoint.get("splits") is not None and checkpoint["splits"] != splits:
            raise ValueError("Checkpoint split does not match the current deterministic split.")
        checkpoint_args = checkpoint.get("args", {})
        if checkpoint_args.get("model") not in (None, args.model):
            raise ValueError("Checkpoint model does not match --model.")
        if checkpoint_args.get("target_mode") not in (None, args.target_mode):
            raise ValueError("Checkpoint target mode does not match --target-mode.")
        history = checkpoint.get("history", [])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        start_epoch = int(checkpoint["epoch"]) + 1

    save_config(
        args,
        run_dir,
        data_dir,
        root_files,
        event_sources,
        splits,
        {name: target_order_counts(events, indices) for name, indices in splits.items()},
    )
    save_history(history, run_dir)

    for epoch in range(start_epoch, args.epochs):
        epoch_metrics = {"epoch": epoch + 1, "lr": optimizer.param_groups[0]["lr"]}
        epoch_metrics.update(
            train_one_epoch(
                model,
                training_events,
                splits["train"],
                training_view_fn,
                optimizer,
                args,
                device,
                epoch,
                logger,
            )
        )
        val_start = time.time()
        val_metrics = evaluate(
            model,
            training_events,
            splits["val"],
            training_view_fn,
            args,
            device,
            "val",
        )
        val_metrics["val_elapsed_sec"] = time.time() - val_start
        epoch_metrics.update(val_metrics)
        history.append(epoch_metrics)
        save_history(history, run_dir)
        plot_history(history, run_dir, title_prefix=args.model)

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
        logger.info(
            "epoch=%03d val_loss=%.5f val_acc=%.4f",
            epoch + 1,
            val_metrics["val_loss"],
            val_metrics["val_accuracy"],
        )

    test_metrics = evaluate(
        model,
        training_events,
        splits["test"],
        training_view_fn,
        args,
        device,
        "test",
    )
    final_metrics = {"best_val_loss": best_val_loss, **test_metrics}
    save_json(run_dir / "final_metrics.json", final_metrics)
    plot_confusion_matrix(
        test_metrics["test_confusion"],
        list(range(len(args.valid_labels))),
        run_dir / "test_hit_origin_confusion_matrix.png",
        f"{args.model} test canonical-y hit classification",
    )
    plot_test_predictions(
        model,
        training_events,
        splits["test"],
        training_view_fn,
        args,
        device,
        run_dir,
    )
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
        "Final test: loss=%.5f hit_acc=%.4f; saved outputs to %s",
        test_metrics["test_loss"],
        test_metrics["test_accuracy"],
        run_dir,
    )


if __name__ == "__main__":
    main()
