"""Inspect a saved hit-classifier run without retraining it."""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_hit_classifier_baseline as training

from ml_ldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    filter_noise_tensor_event,
)
from ml_ldmx.datasets.preprocess import normalize_event_continuous_features
from ml_ldmx.eval.event_diagnostics import select_representative_events
from ml_ldmx.eval.hit_classifier_baseline import collect_event_metrics
from ml_ldmx.io.artifacts import save_json
from ml_ldmx.train.logging import setup_logging
from ml_ldmx.train.utils import resolve_device
from ml_ldmx.viz.training import (
    plot_assignment_ceiling_diagnostics,
    plot_event_accuracy_overview,
    plot_event_diagnostic_correlations,
    plot_shower_separation_profiles,
)


MODEL_CLASSES = {
    "ECalGravNet": training.ECalGravNet,
    "ECalTpadGravNet": training.ECalTpadGravNet,
    "ECalTransformer": training.ECalTransformer,
    "ECalTpadTransformer": training.ECalTpadTransformer,
}
MODEL_VIEWS = {
    "ECalGravNet": training.ecal_gravnet_view,
    "ECalTpadGravNet": training.ecal_tpad_gravnet_view,
    "ECalTransformer": training.ecal_transformer_view,
    "ECalTpadTransformer": training.ecal_tpad_transformer_view,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved maintained hit-classifier run and generate event-level "
            "diagnostics and representative interactive displays without retraining."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path or filename. Defaults to best.pt, then latest.pt.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--event-index",
        type=int,
        action="append",
        default=None,
        help="Inspect a specific saved split event index; repeat for multiple events.",
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=9,
        help="Maximum number of worst/median/best interactive event displays to create.",
    )
    parser.add_argument(
        "--max-inspection-events",
        type=int,
        default=None,
        help="Inspect only the first N saved split indices for a quicker exploratory run.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--event-diagnostic-radius-mm", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    data_group = parser.add_argument_group("relocated data overrides")
    data_group.add_argument("--data-root", type=Path, default=None)
    data_group.add_argument("--processed-dir", type=Path, default=None)
    data_group.add_argument("--processed-cache", type=Path, default=None)
    data_group.add_argument("--processed-cache-root", type=Path, default=None)
    data_group.add_argument(
        "--processed-source",
        action="append",
        nargs=3,
        metavar=("ELECTRON_COUNT", "LABEL", "CACHE_DIR"),
        help="Override one multi-source cache; repeat for each source.",
    )
    data_group.add_argument("--events-per-source", type=int, default=None)
    data_group.add_argument("--shard-cache-size", type=int, default=None)
    return parser.parse_args()


def _load_json(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_checkpoint_path(run_dir, requested_path=None):
    if requested_path is not None:
        requested_path = Path(requested_path)
        candidates = [requested_path]
        if not requested_path.is_absolute():
            candidates.extend(
                [
                    run_dir / requested_path,
                    run_dir / "checkpoints" / requested_path,
                ]
            )
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            "Could not find requested checkpoint. Tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    candidates = [
        run_dir / "checkpoints" / "best.pt",
        run_dir / "checkpoints" / "latest.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Run has neither checkpoints/best.pt nor checkpoints/latest.pt: " + str(run_dir)
    )


def _path_value(value):
    return None if value is None else Path(value)


def _training_args(checkpoint, config, inspection_args):
    stored_args = dict(config)
    stored_args.update(checkpoint.get("args", {}))
    defaults = {
        "model": None,
        "events_per_class": 10,
        "max_events": None,
        "data_root": training.DEFAULT_DATA_ROOT,
        "processed_dir": training.DEFAULT_PROCESSED_DIR,
        "processed_cache": None,
        "processed_cache_root": None,
        "processed_source": None,
        "events_per_source": None,
        "force_sharded_cache": False,
        "allow_incomplete_sharded_cache": False,
        "max_cache_root_files": None,
        "max_events_per_root_file": None,
        "shard_cache_size": 1,
        "valid_labels": list(checkpoint.get("valid_labels", training.VALID_LABELS)),
        "target_mode": "canonical-y",
        "ecal_energy_transform": "raw",
        "tpad_pe_transform": "raw",
        "allow_fewer_events": False,
        "no_progress": True,
        "event_log_every": 0,
        "read_step_size": 500,
        "batch_size": 8,
        "event_diagnostic_radius_mm": 25.0,
    }
    for key, value in defaults.items():
        stored_args.setdefault(key, value)

    for key in (
        "data_root",
        "processed_dir",
        "processed_cache",
        "processed_cache_root",
    ):
        stored_args[key] = _path_value(stored_args.get(key))
    if stored_args.get("processed_source"):
        stored_args["processed_source"] = [
            (int(electron_count), label, Path(cache_dir))
            for electron_count, label, cache_dir in stored_args["processed_source"]
        ]

    source_overrides = [
        inspection_args.processed_dir,
        inspection_args.processed_cache,
        inspection_args.processed_cache_root,
        inspection_args.processed_source,
    ]
    if sum(value is not None for value in source_overrides) > 1:
        raise ValueError(
            "Use only one of --processed-dir, --processed-cache, "
            "--processed-cache-root, or --processed-source."
        )
    if any(value is not None for value in source_overrides):
        stored_args["processed_dir"] = training.DEFAULT_PROCESSED_DIR
        stored_args["processed_cache"] = None
        stored_args["processed_cache_root"] = None
        stored_args["processed_source"] = None
        if inspection_args.processed_dir is not None:
            stored_args["processed_dir"] = inspection_args.processed_dir
        elif inspection_args.processed_cache is not None:
            stored_args["processed_cache"] = inspection_args.processed_cache
        elif inspection_args.processed_cache_root is not None:
            stored_args["processed_cache_root"] = inspection_args.processed_cache_root
        else:
            stored_args["processed_source"] = [
                (int(electron_count), label, Path(cache_dir))
                for electron_count, label, cache_dir in inspection_args.processed_source
            ]
    else:
        resolved_data_dir = config.get("resolved_data_dir")
        if resolved_data_dir:
            resolved_path = Path(resolved_data_dir)
            if resolved_path.exists():
                if stored_args["processed_cache"] is not None:
                    stored_args["processed_cache"] = resolved_path
                elif stored_args["processed_cache_root"] is None and not stored_args["processed_source"]:
                    stored_args["processed_dir"] = resolved_path
                    stored_args["data_root"] = resolved_path

    if inspection_args.data_root is not None:
        stored_args["data_root"] = inspection_args.data_root
    if inspection_args.events_per_source is not None:
        stored_args["events_per_source"] = inspection_args.events_per_source
    if inspection_args.shard_cache_size is not None:
        stored_args["shard_cache_size"] = inspection_args.shard_cache_size
    if inspection_args.batch_size is not None:
        stored_args["batch_size"] = inspection_args.batch_size
    if inspection_args.event_diagnostic_radius_mm is not None:
        stored_args["event_diagnostic_radius_mm"] = inspection_args.event_diagnostic_radius_mm

    stored_args["valid_labels"] = list(checkpoint.get("valid_labels", stored_args["valid_labels"]))
    stored_args["num_diagnostic_event_displays"] = inspection_args.num_events
    stored_args["no_progress"] = True
    return SimpleNamespace(**stored_args)


def _feature_norm_from_checkpoint(checkpoint):
    stored = checkpoint.get("feature_norm")
    if stored is None:
        return None
    return {
        "first_continuous_col": int(stored["first_continuous_col"]),
        "mean": torch.as_tensor(stored["mean"], dtype=torch.float32),
        "std": torch.as_tensor(stored["std"], dtype=torch.float32),
    }


def restore_event_preprocessing(events, checkpoint, args):
    """Apply the checkpoint's targets and fixed feature normalization."""
    feature_norm = _feature_norm_from_checkpoint(checkpoint)

    def transform(event):
        event = filter_noise_tensor_event(event)
        event = apply_variable_count_target_mode(
            event,
            valid_labels=tuple(args.valid_labels),
            target_mode=args.target_mode,
            max_electrons=len(args.valid_labels),
        )
        if feature_norm is not None:
            event = normalize_event_continuous_features(event, feature_norm)
        return event

    if training._is_sharded_events(events):
        events.set_event_transform(transform)
        return events
    if not isinstance(events, list):
        raise TypeError(f"Unsupported eager event container: {type(events).__name__}.")
    for event_idx, event in enumerate(events):
        events[event_idx] = transform(event)
    return events


def restore_model(checkpoint, args, device):
    model_name = args.model
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported checkpoint model: {model_name!r}.")
    model_kwargs = checkpoint.get("model_kwargs")
    if not model_kwargs:
        raise ValueError("Checkpoint does not contain model_kwargs required for inspection.")
    model = MODEL_CLASSES[model_name](**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, MODEL_VIEWS[model_name]


def validate_saved_split(events, checkpoint, config, split_name):
    splits = checkpoint.get("splits")
    if not isinstance(splits, dict) or split_name not in splits:
        raise ValueError(f"Checkpoint does not contain a saved {split_name!r} split.")
    indices = [int(index) for index in splits[split_name]]
    if indices and (min(indices) < 0 or max(indices) >= len(events)):
        raise ValueError(
            f"Saved {split_name} split requires event index {max(indices)}, "
            f"but the loaded dataset has {len(events)} events. Use a relocated data override."
        )
    expected_events = config.get("num_loaded_events")
    if expected_events is not None and int(expected_events) != len(events):
        raise ValueError(
            f"Saved run used {expected_events} events, but the loaded dataset has {len(events)}. "
            "Point the inspector at the same processed dataset."
        )
    return indices


def summarize_records(records):
    accuracies = [float(record["accuracy"]) for record in records if record.get("accuracy") is not None]
    losses = [float(record["loss"]) for record in records if record.get("loss") is not None]
    correct_hits = sum(int(record.get("correct_hits", 0)) for record in records)
    num_hits = sum(int(record.get("num_hits", 0)) for record in records)
    return {
        "num_events": len(records),
        "num_hits": num_hits,
        "correct_hits": correct_hits,
        "hit_accuracy": correct_hits / num_hits if num_hits else None,
        "mean_event_accuracy": statistics.fmean(accuracies) if accuracies else None,
        "median_event_accuracy": statistics.median(accuracies) if accuracies else None,
        "mean_event_loss": statistics.fmean(losses) if losses else None,
    }


def main():
    inspection_args = parse_args()
    if inspection_args.num_events < 0:
        raise ValueError("--num-events must be non-negative.")
    if inspection_args.max_inspection_events is not None and inspection_args.max_inspection_events <= 0:
        raise ValueError("--max-inspection-events must be positive when provided.")

    run_dir = inspection_args.run_dir.resolve()
    checkpoint_path = resolve_checkpoint_path(run_dir, inspection_args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = _load_json(run_dir / "config.json")
    args = _training_args(checkpoint, config, inspection_args)

    output_dir = inspection_args.output_dir
    if output_dir is None:
        output_dir = run_dir / "inspection" / checkpoint_path.stem / inspection_args.split
    output_dir = output_dir.resolve()
    logger = setup_logging(
        output_dir,
        logger_name="inspect_hit_classifier_run",
        log_filename="inspection.log",
    )
    device = resolve_device(inspection_args.device, logger)
    logger.info("Inspecting run: %s", run_dir)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Model: %s; split: %s; device: %s", args.model, inspection_args.split, device)

    events, _event_sources, data_dir, _root_files = training.load_events(args, logger)
    saved_split_indices = validate_saved_split(
        events,
        checkpoint,
        config,
        inspection_args.split,
    )
    split_indices = saved_split_indices
    if inspection_args.event_index:
        requested_indices = list(dict.fromkeys(int(index) for index in inspection_args.event_index))
        missing_indices = sorted(set(requested_indices) - set(saved_split_indices))
        if missing_indices:
            raise ValueError(
                f"Requested event indices are not in the saved {inspection_args.split!r} split: "
                f"{missing_indices}."
            )
        split_indices = requested_indices
    if inspection_args.max_inspection_events is not None:
        split_indices = split_indices[: inspection_args.max_inspection_events]
    if not split_indices:
        raise ValueError(f"Saved {inspection_args.split!r} split contains no events to inspect.")

    events = restore_event_preprocessing(events, checkpoint, args)
    model, view_fn = restore_model(checkpoint, args, device)
    records = collect_event_metrics(model, events, split_indices, view_fn, args, device)

    record_paths = training.save_event_accuracy_records(
        records,
        output_dir,
        inspection_args.split,
    )
    accuracy_plot_path = output_dir / f"{inspection_args.split}_event_accuracy_overview.png"
    plot_event_accuracy_overview(
        records,
        accuracy_plot_path,
        f"{args.model} {inspection_args.split} event hit accuracy",
    )
    correlation_plot_path = output_dir / f"{inspection_args.split}_event_diagnostic_correlations.png"
    plot_event_diagnostic_correlations(
        records,
        correlation_plot_path,
        f"{args.model} {inspection_args.split} event diagnostics",
    )
    ceiling_plot_path = output_dir / f"{inspection_args.split}_assignment_ceiling_diagnostics.png"
    plot_assignment_ceiling_diagnostics(
        records,
        ceiling_plot_path,
        f"{args.model} {inspection_args.split} assignment-ceiling diagnostics",
    )
    separation_plot_path = output_dir / f"{inspection_args.split}_shower_separation_profiles.png"
    plot_shower_separation_profiles(
        records,
        separation_plot_path,
        f"{args.model} {inspection_args.split} accuracy versus shower separation",
    )

    if inspection_args.event_index:
        selection = {
            "metric": "requested_event_idx",
            "requested": records,
            "worst": [],
            "median": [],
            "best": [],
        }
    else:
        per_group = (
            max(1, math.ceil(inspection_args.num_events / 3))
            if inspection_args.num_events
            else 0
        )
        selection = select_representative_events(
            records,
            limit_per_group=per_group,
            metric="accuracy",
        )
    selection_path = output_dir / f"{inspection_args.split}_representative_events.json"
    save_json(selection_path, selection)
    display_paths = training.plot_representative_predictions(
        model,
        events,
        view_fn,
        selection,
        args,
        device,
        output_dir / f"{inspection_args.split}_representative_events",
        logger,
        split_name=inspection_args.split,
    )

    summary = summarize_records(records)
    generated_paths = [record_paths["json"], record_paths["csv"], accuracy_plot_path, selection_path]
    if correlation_plot_path.exists():
        generated_paths.append(correlation_plot_path)
    if ceiling_plot_path.exists():
        generated_paths.append(ceiling_plot_path)
    if separation_plot_path.exists():
        generated_paths.append(separation_plot_path)
    generated_paths.extend(display_paths)
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model": args.model,
        "split": inspection_args.split,
        "data_dir": str(data_dir),
        "used_saved_feature_normalization": checkpoint.get("feature_norm") is not None,
        "event_diagnostic_radius_mm": float(args.event_diagnostic_radius_mm),
        "limited_to_first_split_events": inspection_args.max_inspection_events,
        "requested_event_indices": inspection_args.event_index,
        "summary": summary,
        "generated_files": [str(path.relative_to(output_dir)) for path in generated_paths],
    }
    manifest_path = output_dir / "inspection_manifest.json"
    save_json(manifest_path, manifest)
    accuracy_text = (
        f"{summary['hit_accuracy']:.4f}"
        if summary["hit_accuracy"] is not None
        else "unavailable"
    )
    logger.info(
        "Inspected %s events (%s hits), hit accuracy %s.",
        summary["num_events"],
        summary["num_hits"],
        accuracy_text,
    )
    logger.info("Saved inspection bundle to %s", output_dir)


if __name__ == "__main__":
    main()
