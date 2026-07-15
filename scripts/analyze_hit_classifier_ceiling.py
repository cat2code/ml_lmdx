"""Analyze label binding, truth ambiguity, and TPad reliance for a saved run."""

import argparse
import statistics
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = Path(__file__).resolve().parent
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import inspect_hit_classifier_run as inspection
import train_hit_classifier_baseline as training

from ml_ldmx.eval.hit_classifier_baseline import collect_event_metrics
from ml_ldmx.io.artifacts import save_json
from ml_ldmx.train.logging import setup_logging
from ml_ldmx.train.utils import resolve_device
from ml_ldmx.viz.training import (
    plot_assignment_ceiling_diagnostics,
    plot_tpad_ablation_comparison,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure assignment ceilings and paired TPad-token ablation for a saved "
            "hit-classifier checkpoint."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-inspection-events", type=int, default=None)
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
    )
    data_group.add_argument("--events-per-source", type=int, default=None)
    data_group.add_argument("--shard-cache-size", type=int, default=None)
    args = parser.parse_args()
    args.num_events = 0
    return args


def remove_tpad_tokens(view):
    """Return one model view with all TPad tokens removed and hit fields unchanged."""
    if "tpad_mask" not in view:
        raise ValueError("TPad ablation requires a combined ECal+TPad model view.")
    tpad_mask = view["tpad_mask"].to(dtype=bool)
    if tpad_mask.shape != (view["x"].shape[0],):
        raise ValueError("view['tpad_mask'] must align with view['x'].")
    keep = ~tpad_mask
    ablated = dict(view)
    ablated["x"] = view["x"][keep]
    ablated["ecal_mask"] = view["ecal_mask"][keep]
    ablated["tpad_mask"] = view["tpad_mask"][keep]
    if "tpad" in view:
        ablated["tpad"] = view["tpad"][:0]
    if "tpad_raw_pe" in view:
        ablated["tpad_raw_pe"] = view["tpad_raw_pe"][:0]
    return ablated


def _hit_weighted_summary(records):
    hits = sum(int(record.get("num_hits", 0)) for record in records)
    correct = sum(int(record.get("correct_hits", 0)) for record in records)
    return {
        "num_events": len(records),
        "num_hits": hits,
        "hit_accuracy": None if hits == 0 else correct / hits,
        "mean_event_accuracy": (
            None
            if not records
            else statistics.fmean(float(record["accuracy"]) for record in records)
        ),
    }


def summarize_paired_tpad_ablation(reference_records, ablated_records):
    """Return JSON-safe paired aggregate metrics for TPad removal."""
    reference = {int(record["event_idx"]): record for record in reference_records}
    ablated = {int(record["event_idx"]): record for record in ablated_records}
    if set(reference) != set(ablated):
        raise ValueError("Reference and TPad-ablated records must contain identical event sets.")
    event_indices = sorted(reference)
    pairs = [(reference[index], ablated[index]) for index in event_indices]
    for original, removed in pairs:
        if int(original["num_hits"]) != int(removed["num_hits"]):
            raise ValueError(f"TPad ablation changed ECal hit count for event {original['event_idx']}.")

    original_summary = _hit_weighted_summary([pair[0] for pair in pairs])
    removed_summary = _hit_weighted_summary([pair[1] for pair in pairs])
    event_gains = [float(original["accuracy"]) - float(removed["accuracy"]) for original, removed in pairs]
    summary = {
        "reference": original_summary,
        "tpad_ablated": removed_summary,
        "tpad_hit_accuracy_gain": (
            None
            if original_summary["hit_accuracy"] is None
            else original_summary["hit_accuracy"] - removed_summary["hit_accuracy"]
        ),
        "mean_event_accuracy_gain": statistics.fmean(event_gains) if event_gains else None,
        "median_event_accuracy_gain": statistics.median(event_gains) if event_gains else None,
        "events_helped_by_tpad": sum(gain > 1e-12 for gain in event_gains),
        "events_hurt_by_tpad": sum(gain < -1e-12 for gain in event_gains),
        "events_unchanged_by_tpad": sum(abs(gain) <= 1e-12 for gain in event_gains),
    }

    by_completeness = {}
    for key, predicate in (
        ("complete", lambda record: bool(record.get("has_complete_tpad_context"))),
        ("incomplete", lambda record: not bool(record.get("has_complete_tpad_context"))),
    ):
        selected = [pair for pair in pairs if predicate(pair[0])]
        if selected:
            by_completeness[key] = {
                "reference": _hit_weighted_summary([pair[0] for pair in selected]),
                "tpad_ablated": _hit_weighted_summary([pair[1] for pair in selected]),
            }
            by_completeness[key]["tpad_hit_accuracy_gain"] = (
                by_completeness[key]["reference"]["hit_accuracy"]
                - by_completeness[key]["tpad_ablated"]["hit_accuracy"]
            )
    summary["by_tpad_completeness"] = by_completeness
    return summary


def main():
    args = parse_args()
    if args.max_inspection_events is not None and args.max_inspection_events <= 0:
        raise ValueError("--max-inspection-events must be positive when provided.")

    run_dir = args.run_dir.resolve()
    checkpoint_path = inspection.resolve_checkpoint_path(run_dir, args.checkpoint)
    checkpoint = inspection._load_checkpoint(checkpoint_path)
    config = inspection._load_json(run_dir / "config.json")
    training_args = inspection._training_args(checkpoint, config, args)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = run_dir / "ceiling_analysis" / checkpoint_path.stem / args.split
    output_dir = output_dir.resolve()
    logger = setup_logging(
        output_dir,
        logger_name="analyze_hit_classifier_ceiling",
        log_filename="ceiling_analysis.log",
    )
    device = resolve_device(args.device, logger)

    events, _sources, data_dir, _root_files = training.load_events(training_args, logger)
    split_indices = inspection.validate_saved_split(events, checkpoint, config, args.split)
    if args.max_inspection_events is not None:
        split_indices = split_indices[: args.max_inspection_events]
    events = inspection.restore_event_preprocessing(events, checkpoint, training_args)
    model, view_fn = inspection.restore_model(checkpoint, training_args, device)

    reference_records = collect_event_metrics(
        model,
        events,
        split_indices,
        view_fn,
        training_args,
        device,
    )

    def ablated_view_fn(event):
        return remove_tpad_tokens(view_fn(event))

    ablated_records = collect_event_metrics(
        model,
        events,
        split_indices,
        ablated_view_fn,
        training_args,
        device,
    )

    reference_paths = training.save_event_accuracy_records(reference_records, output_dir, "reference")
    ablated_paths = training.save_event_accuracy_records(ablated_records, output_dir, "tpad_ablated")
    ceiling_plot_path = output_dir / "reference_assignment_ceiling_diagnostics.png"
    plot_assignment_ceiling_diagnostics(
        reference_records,
        ceiling_plot_path,
        f"{training_args.model} {args.split} assignment-ceiling diagnostics",
    )
    ablation_plot_path = output_dir / "tpad_ablation.png"
    plot_tpad_ablation_comparison(
        reference_records,
        ablated_records,
        ablation_plot_path,
        f"{training_args.model} {args.split} paired TPad ablation",
    )

    summary = summarize_paired_tpad_ablation(reference_records, ablated_records)
    summary_path = output_dir / "ceiling_summary.json"
    save_json(summary_path, summary)
    generated = [
        reference_paths["json"],
        reference_paths["csv"],
        ablated_paths["json"],
        ablated_paths["csv"],
        ceiling_plot_path,
        ablation_plot_path,
        summary_path,
    ]
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "model": training_args.model,
        "split": args.split,
        "num_events": len(reference_records),
        "data_dir": str(data_dir),
        "limited_to_first_split_events": args.max_inspection_events,
        "summary": summary,
        "generated_files": [str(path.relative_to(output_dir)) for path in generated],
    }
    manifest_path = output_dir / "ceiling_manifest.json"
    save_json(manifest_path, manifest)
    logger.info("Saved paired ceiling analysis to %s", output_dir)
    logger.info("TPad hit-accuracy gain: %s", summary["tpad_hit_accuracy_gain"])


if __name__ == "__main__":
    main()
