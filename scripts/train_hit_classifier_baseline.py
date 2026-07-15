"""Train one maintained ECal hit-origin classification baseline."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from ml_ldmx.datasets.cached_views import CachedEventViewDataset
from ml_ldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    apply_variable_count_target_mode_to_events,
    filter_noise_tensor_event,
    load_or_create_sharded_tensor_events,
    load_multi_sharded_tensor_events,
    load_processed_or_grouped_root_tensor_events,
)
from ml_ldmx.datasets.ecal_tpad_shards import MultiShardedECalTpadDataset, ShardedECalTpadDataset
from ml_ldmx.datasets.model_views import (
    ecal_gravnet_view,
    ecal_tpad_gravnet_view,
    ecal_tpad_transformer_view,
    ecal_transformer_view,
)
from ml_ldmx.datasets.preprocess import (
    fit_continuous_feature_normalization,
    normalize_continuous_features,
    normalize_event_continuous_features,
)
from ml_ldmx.datasets.stats import count_classes, target_order_counts
from ml_ldmx.eval.event_diagnostics import select_representative_events
from ml_ldmx.eval.hit_classifier_baseline import collect_event_metrics, evaluate
from ml_ldmx.io.artifacts import save_config, save_history, save_json
from ml_ldmx.models import ECalGravNet, ECalTpadGravNet, ECalTpadTransformer, ECalTransformer
from ml_ldmx.train.checkpoints import load_checkpoint, save_checkpoint
from ml_ldmx.train.hit_classifier_baseline import compute_event_losses, train_one_epoch
from ml_ldmx.train.logging import setup_logging
from ml_ldmx.train.modeling import count_trainable_parameters
from ml_ldmx.train.paths import resolve_existing_path, resolve_run_dir
from ml_ldmx.train.progress import make_progress
from ml_ldmx.train.run_overview import (
    build_run_overview,
    log_run_overview,
    save_run_overview,
)
from ml_ldmx.train.splits import deterministic_split
from ml_ldmx.train.utils import resolve_device
from ml_ldmx.viz.ecal import (
    plot_ecal_hit_prediction_errors_3d,
    plot_ecal_hit_prediction_errors_3d_interactive,
)
from ml_ldmx.viz.training import (
    plot_assignment_ceiling_diagnostics,
    plot_confusion_matrix,
    plot_event_accuracy_overview,
    plot_event_diagnostic_correlations,
    plot_history,
    plot_shower_separation_profiles,
)


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
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=(
            "Existing processed dataset. If the directory has sharded manifest/index files, "
            "it is loaded lazily as shards; otherwise legacy event_*.pt files are loaded."
        ),
    )
    parser.add_argument(
        "--processed-cache",
        type=Path,
        default=None,
        help=(
            "ML-ready sharded cache to reuse or create from --data-root "
            "(one ROOT file per shard; recommended for large training runs)."
        ),
    )
    parser.add_argument(
        "--processed-cache-root",
        type=Path,
        default=None,
        help=(
            "Directory containing separate 2e/events and 3e/events sharded caches; "
            "recommended for production jobs prepared by scripts/slurm."
        ),
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
    parser.add_argument(
        "--shard-cache-size",
        type=int,
        default=1,
        help=(
            "Number of recently loaded processed shards retained in CPU RAM. "
            "Use 1 for memory-constrained local runs; raise this on the cluster when RAM allows."
        ),
    )
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
        "--ecal-energy-transform",
        choices=("raw", "log1p"),
        default="raw",
        help=(
            "Transform the reconstructed ECal energy input feature during ROOT preprocessing. "
            "Existing processed caches must record the same transform."
        ),
    )
    parser.add_argument(
        "--tpad-pe-transform",
        choices=("raw", "log1p"),
        default="raw",
        help=(
            "Transform the TriggerPadTracks pe input feature during ROOT preprocessing. "
            "Existing processed caches must record the same transform."
        ),
    )
    parser.add_argument(
        "--cache-model-views",
        action="store_true",
        help=(
            "Cache selected adapter views for reuse across epochs. Local in-memory datasets "
            "are materialized eagerly; sharded datasets use a bounded lazy LRU by default."
        ),
    )
    parser.add_argument(
        "--model-view-cache-size",
        type=int,
        default=None,
        help=(
            "Maximum cached model views when --cache-model-views is enabled. "
            "For sharded datasets, omit to cache roughly --shard-cache-size shard(s); "
            "use 0 only when you intentionally want an unbounded/all-event view cache."
        ),
    )
    parser.add_argument(
        "--allow-small-split",
        action="store_true",
        help="Permit non-empty deterministic splits below 20 events for smoke validation.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--num-ecal-plots", type=int, default=2)
    parser.add_argument(
        "--num-diagnostic-event-displays",
        type=int,
        default=3,
        help=(
            "Number of validation representative events to render after training. "
            "The default gives one worst, one median-like, and one best event."
        ),
    )
    parser.add_argument(
        "--event-diagnostic-radius-mm",
        type=float,
        default=25.0,
        help="Centroid-neighborhood radius used in per-event shower-overlap diagnostics.",
    )
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
    if args.model_view_cache_size is not None and args.model_view_cache_size < 0:
        raise ValueError("--model-view-cache-size must be non-negative when provided.")
    if args.checkpoint_every <= 0 or args.read_step_size < 0:
        raise ValueError("--checkpoint-every must be positive and --read-step-size non-negative.")
    if args.num_ecal_plots < 0 or args.num_diagnostic_event_displays < 0:
        raise ValueError("--num-ecal-plots and --num-diagnostic-event-displays must be non-negative.")
    if args.event_diagnostic_radius_mm <= 0:
        raise ValueError("--event-diagnostic-radius-mm must be positive.")
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
            ecal_energy_transform=args.ecal_energy_transform,
            tpad_pe_transform=args.tpad_pe_transform,
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
            ecal_energy_transform=args.ecal_energy_transform,
            tpad_pe_transform=args.tpad_pe_transform,
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
        shard_cache_size=args.shard_cache_size,
        allow_incomplete_sharded_cache=args.allow_incomplete_sharded_cache,
        ecal_energy_transform=args.ecal_energy_transform,
        tpad_pe_transform=args.tpad_pe_transform,
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


def _is_sharded_events(events):
    return isinstance(events, (ShardedECalTpadDataset, MultiShardedECalTpadDataset))


def _largest_reachable_shards(dataset, shard_cache_size):
    counts = sorted(dataset.shard_event_counts, reverse=True)
    return sum(counts[: int(shard_cache_size)])


def _default_model_view_cache_size(events, args):
    if args.model_view_cache_size == 0:
        return None
    if args.model_view_cache_size is not None:
        return int(args.model_view_cache_size)
    if isinstance(events, MultiShardedECalTpadDataset):
        return max(
            int(args.batch_size),
            sum(
                _largest_reachable_shards(source["dataset"], args.shard_cache_size)
                for source in events.sources
            ),
        )
    if isinstance(events, ShardedECalTpadDataset):
        return max(int(args.batch_size), _largest_reachable_shards(events, args.shard_cache_size))
    return None


def prepare_training_views(events, view_fn, args, logger):
    if not args.cache_model_views:
        logger.info(
            "Deriving model-specific views on demand; pass --cache-model-views to reuse them."
        )
        return events, view_fn, {"enabled": False, "policy": "on_demand"}

    if _is_sharded_events(events):
        max_cache_events = _default_model_view_cache_size(events, args)
        cached_events = CachedEventViewDataset(
            events,
            view_fn,
            max_cache_events=max_cache_events,
        )
        logger.info(
            "Using lazy model-view cache for sharded data: max_cache_events=%s. "
            "Omit --model-view-cache-size to cache about --shard-cache-size shard(s); "
            "set it to 0 only for an intentional unbounded cache.",
            "unbounded" if max_cache_events is None else max_cache_events,
        )
        return (
            cached_events,
            None,
            {
                "enabled": True,
                "policy": "lazy_lru",
                "max_cache_events": max_cache_events,
            },
        )

    if args.model_view_cache_size not in (None, 0):
        cached_events = CachedEventViewDataset(
            events,
            view_fn,
            max_cache_events=args.model_view_cache_size,
        )
        logger.info(
            "Using bounded lazy model-view cache: max_cache_events=%s.",
            args.model_view_cache_size,
        )
        return (
            cached_events,
            None,
            {
                "enabled": True,
                "policy": "lazy_lru",
                "max_cache_events": int(args.model_view_cache_size),
            },
        )

    training_events = [view_fn(event) for event in events]
    logger.info("Prepared %s model-specific event views once for epoch reuse.", len(training_events))
    return (
        training_events,
        None,
        {
            "enabled": True,
            "policy": "eager_all_events",
            "max_cache_events": len(training_events),
        },
    )


def save_event_accuracy_records(records, run_dir, split_name):
    json_path = run_dir / f"{split_name}_event_accuracy.json"
    csv_path = run_dir / f"{split_name}_event_accuracy.csv"
    save_json(json_path, records)
    base_fieldnames = [
        "split_position",
        "event_idx",
        "num_hits",
        "correct_hits",
        "incorrect_hits",
        "accuracy",
        "loss",
        "source_file",
        "source_entry",
        "source_label",
        "electron_count",
        "target_label_order",
    ]
    extra_fieldnames = sorted(
        {
            key
            for record in records
            for key in record.keys()
            if key not in base_fieldnames
        }
    )
    fieldnames = base_fieldnames + extra_fieldnames

    def csv_value(value):
        if isinstance(value, (list, dict)):
            return json.dumps(value)
        return value

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: csv_value(record.get(key)) for key in fieldnames})
    return {"json": json_path, "csv": csv_path}


def log_worst_event_accuracies(logger, records, split_name, limit=8):
    valid_records = [record for record in records if record.get("accuracy") is not None]
    if not valid_records:
        logger.info("No %s event-accuracy records were available.", split_name)
        return
    worst = sorted(
        valid_records,
        key=lambda record: (
            float(record["accuracy"]),
            -int(record.get("incorrect_hits", 0)),
            int(record["event_idx"]),
        ),
    )[:limit]
    summary = [
        (
            record["event_idx"],
            record["accuracy"],
            record["incorrect_hits"],
            record["num_hits"],
            record.get("source_file"),
        )
        for record in worst
    ]
    logger.info("Lowest %s event accuracies: %s", split_name, summary)


def _representative_display_records(selection, max_total):
    if max_total <= 0:
        return []
    groups = tuple(
        group
        for group in ("requested", "worst", "median", "best")
        if group in selection
    )
    group_positions = {group: 0 for group in groups}
    selected = []
    seen_event_indices = set()
    while len(selected) < max_total:
        made_progress = False
        for group in groups:
            records = selection.get(group, [])
            position = group_positions[group]
            if position >= len(records):
                continue
            group_positions[group] += 1
            made_progress = True
            record = records[position]
            event_idx = int(record["event_idx"])
            if event_idx in seen_event_indices:
                continue
            selected.append((group, record))
            seen_event_indices.add(event_idx)
            if len(selected) >= max_total:
                break
        if not made_progress:
            break
    return selected


def _prediction_confidence_and_entropy(logits):
    logits = logits.detach().cpu()
    probabilities = F.softmax(logits, dim=1)
    confidence = probabilities.max(dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    if probabilities.shape[1] > 1:
        entropy = entropy / torch.log(torch.tensor(float(probabilities.shape[1])))
    return confidence, entropy


def _predicted_display_labels(view, pred_class):
    if "origin_id_y" not in view:
        return pred_class.detach().cpu()
    target_label_order = view.get("target_label_order")
    if target_label_order is None:
        return pred_class.detach().cpu()
    if isinstance(target_label_order, torch.Tensor):
        target_label_order = target_label_order.detach().cpu().tolist()
    target_label_order = [int(label) for label in target_label_order]
    pred_class = pred_class.detach().cpu().to(dtype=torch.long)
    mapped = []
    for value in pred_class.tolist():
        value = int(value)
        mapped.append(target_label_order[value] if 0 <= value < len(target_label_order) else value)
    return torch.tensor(mapped, dtype=torch.long)


def plot_representative_predictions(
    model,
    events_or_views,
    view_fn,
    selection,
    args,
    device,
    output_dir,
    logger,
    split_name="val",
):
    display_records = _representative_display_records(
        selection,
        max_total=args.num_diagnostic_event_displays,
    )
    if not display_records:
        return []

    split_name = str(split_name)
    split_title = {"val": "validation"}.get(split_name, split_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    model.eval()
    for group, record in display_records:
        event_idx = int(record["event_idx"])
        losses = compute_event_losses(model, events_or_views[event_idx], view_fn, device)
        view = losses["view"]
        logits = losses.get("supervised_logits")
        confidence, entropy = _prediction_confidence_and_entropy(logits)
        color_labels = view.get("origin_id_y", losses["true_class"]).detach().cpu()
        labels = (
            list(args.valid_labels)
            if "origin_id_y" in view
            else list(range(len(args.valid_labels)))
        )
        stem = f"{split_name}_{group}_event_{event_idx:04d}"
        title = (
            f"{args.model} {split_title} {group} event {event_idx}, "
            f"accuracy={record.get('accuracy'):.3f}"
        )
        png_path = output_dir / f"{stem}_prediction_errors.png"
        plot_ecal_hit_prediction_errors_3d(
            view["ecal_pos"].detach().cpu(),
            losses["true_class"].detach().cpu(),
            losses["pred_class"].detach().cpu(),
            output_path=png_path,
            title=title,
            labels=labels,
            color_labels=color_labels,
            legend_title="true origin_id / marker",
        )
        saved_paths.append(png_path)

        html_path = output_dir / f"{stem}_interactive.html"
        try:
            plot_ecal_hit_prediction_errors_3d_interactive(
                view["ecal_pos"].detach().cpu(),
                color_labels,
                _predicted_display_labels(view, losses["pred_class"]),
                output_path=html_path,
                title=title,
                energy=view.get("ecal_raw_energy", view.get("ecal_input_energy")),
                confidence=confidence,
                entropy=entropy,
            )
        except ImportError as exc:
            logger.warning(
                "Skipping interactive %s display for event %s: %s",
                split_title,
                event_idx,
                exc,
            )
        else:
            saved_paths.append(html_path)
    return saved_paths


@torch.no_grad()
def plot_test_predictions(model, events_or_views, test_indices, view_fn, args, device, run_dir):
    model.eval()
    for event_idx in test_indices[: args.num_ecal_plots]:
        losses = compute_event_losses(model, events_or_views[event_idx], view_fn, device)
        view = losses["view"]
        color_labels = view.get("origin_id_y", losses["true_class"]).detach().cpu()
        labels = (
            list(args.valid_labels)
            if "origin_id_y" in view
            else list(range(len(args.valid_labels)))
        )
        plot_ecal_hit_prediction_errors_3d(
            view["ecal_pos"].detach().cpu(),
            losses["true_class"].detach().cpu(),
            losses["pred_class"].detach().cpu(),
            output_path=run_dir / f"test_ecal_event_{event_idx:04d}_prediction_errors.png",
            title=f"{args.model} test event {event_idx}, canonical-y hit prediction errors",
            labels=labels,
            color_labels=color_labels,
            legend_title="true origin_id / marker",
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
    class_counts_by_split = {
        name: count_classes(events, indices)
        for name, indices in splits.items()
    }
    logger.info("Training canonical class counts: %s", class_counts_by_split["train"])

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
    logger.info("Input feature dimension: %s", input_dim)
    training_events, training_view_fn, view_cache_info = prepare_training_views(
        events,
        view_fn,
        args,
        logger,
    )
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

    target_order_counts_by_split = {
        name: target_order_counts(events, indices)
        for name, indices in splits.items()
    }
    save_config(
        args,
        run_dir,
        data_dir,
        root_files,
        event_sources,
        splits,
        target_order_counts_by_split,
    )
    run_overview = build_run_overview(
        args=args,
        run_dir=run_dir,
        model=model,
        model_kwargs=model_kwargs,
        input_dim=input_dim,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        feature_norm=feature_norm,
        data_dir=data_dir,
        event_sources=event_sources,
        splits=splits,
        class_counts_by_split=class_counts_by_split,
        target_order_counts_by_split=target_order_counts_by_split,
        view_fn=view_fn,
        training_view_fn=training_view_fn,
        view_cache_info=view_cache_info,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )
    run_overview_paths = save_run_overview(run_dir, run_overview)
    log_run_overview(logger, run_overview, run_overview_paths)
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

    val_event_records = collect_event_metrics(
        model,
        training_events,
        splits["val"],
        training_view_fn,
        args,
        device,
    )
    val_event_paths = save_event_accuracy_records(val_event_records, run_dir, "val")
    plot_event_accuracy_overview(
        val_event_records,
        run_dir / "val_event_accuracy_overview.png",
        f"{args.model} validation event hit accuracy",
    )
    val_diagnostic_plot_path = run_dir / "val_event_diagnostic_correlations.png"
    plot_event_diagnostic_correlations(
        val_event_records,
        val_diagnostic_plot_path,
        f"{args.model} validation event diagnostics",
    )
    val_ceiling_plot_path = run_dir / "val_assignment_ceiling_diagnostics.png"
    plot_assignment_ceiling_diagnostics(
        val_event_records,
        val_ceiling_plot_path,
        f"{args.model} validation assignment-ceiling diagnostics",
    )
    val_separation_plot_path = run_dir / "val_shower_separation_profiles.png"
    plot_shower_separation_profiles(
        val_event_records,
        val_separation_plot_path,
        f"{args.model} validation accuracy versus shower separation",
    )
    representative_selection = select_representative_events(
        val_event_records,
        limit_per_group=max(3, args.num_diagnostic_event_displays),
        metric="accuracy",
    )
    save_json(run_dir / "val_representative_events.json", representative_selection)
    representative_display_paths = plot_representative_predictions(
        model,
        training_events,
        training_view_fn,
        representative_selection,
        args,
        device,
        run_dir / "val_representative_events",
        logger,
        split_name="val",
    )
    validation_plot_paths = [run_dir / "val_event_accuracy_overview.png"]
    if val_diagnostic_plot_path.exists():
        validation_plot_paths.append(val_diagnostic_plot_path)
    else:
        logger.info(
            "Skipped validation diagnostic-correlation plot; need at least two finite events per panel."
        )
    if val_ceiling_plot_path.exists():
        validation_plot_paths.append(val_ceiling_plot_path)
    else:
        logger.info("Skipped validation assignment-ceiling plot; required diagnostics are unavailable.")
    if val_separation_plot_path.exists():
        validation_plot_paths.append(val_separation_plot_path)
    else:
        logger.info(
            "Skipped validation shower-separation profiles; need finite separation values "
            "for at least one event multiplicity."
        )
    logger.info(
        "Saved validation event diagnostics: %s, %s, plots=%s, representative displays=%s",
        val_event_paths["json"],
        val_event_paths["csv"],
        validation_plot_paths,
        representative_display_paths,
    )
    log_worst_event_accuracies(logger, val_event_records, "validation")

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
    if hasattr(events, "cache_info"):
        logger.info("Final shard cache state: %s", events.cache_info())
    if hasattr(training_events, "cache_info"):
        logger.info("Final model-view cache state: %s", training_events.cache_info())


if __name__ == "__main__":
    main()
