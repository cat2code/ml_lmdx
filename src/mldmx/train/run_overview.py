"""Run overview artifacts for reproducible training runs."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys

import torch

from mldmx.io.artifacts import save_json
from mldmx.io.root_files import root_file_sort_key


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def _format_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _event_source_files(event_sources):
    if hasattr(event_sources, "source_files"):
        return list(event_sources.source_files)
    return [source["file"] for source in event_sources]


def _parameter_overview(model):
    parameters = []
    total = 0
    trainable = 0
    for name, parameter in model.named_parameters():
        numel = int(parameter.numel())
        total += numel
        if parameter.requires_grad:
            trainable += numel
        parameters.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": numel,
                "trainable": bool(parameter.requires_grad),
                "dtype": str(parameter.dtype),
            }
        )

    buffers = [
        {
            "name": name,
            "shape": list(buffer.shape),
            "numel": int(buffer.numel()),
            "dtype": str(buffer.dtype),
        }
        for name, buffer in model.named_buffers()
    ]
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "parameter_tensors": parameters,
        "buffers": buffers,
    }


def _optimizer_overview(optimizer):
    def clean_group(group):
        return {
            key: _jsonable(value)
            for key, value in group.items()
            if key != "params"
        }

    return {
        "class_name": optimizer.__class__.__name__,
        "module": optimizer.__class__.__module__,
        "defaults": clean_group(optimizer.defaults),
        "param_groups": [clean_group(group) for group in optimizer.param_groups],
    }


def _feature_norm_overview(feature_norm):
    if feature_norm is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "first_continuous_col": int(feature_norm["first_continuous_col"]),
        "mean": _jsonable(feature_norm["mean"]),
        "std": _jsonable(feature_norm["std"]),
    }


def build_run_overview(
    *,
    args,
    run_dir,
    model,
    model_kwargs,
    input_dim,
    optimizer,
    scheduler,
    device,
    feature_norm,
    data_dir,
    event_sources,
    splits,
    class_counts_by_split,
    target_order_counts_by_split,
    view_fn,
    training_view_fn,
    view_cache_info,
    start_epoch,
    best_val_loss,
):
    """Build a structured, serializable overview of one training run."""
    source_files = _event_source_files(event_sources)
    parameter_summary = _parameter_overview(model)
    resolved_device = str(device)
    script_path = Path(sys.argv[0]).resolve() if sys.argv else None
    overview = {
        "run": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(run_dir),
            "script": str(script_path) if script_path is not None else None,
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        },
        "model": {
            "requested_name": args.model,
            "class_name": model.__class__.__name__,
            "module": model.__class__.__module__,
            "input_dim": int(input_dim),
            "output_dim": int(len(args.valid_labels)),
            "constructor_kwargs": _jsonable(model_kwargs),
            "architecture": str(model),
            **parameter_summary,
        },
        "training": {
            "start_epoch": int(start_epoch),
            "requested_epochs": int(args.epochs),
            "remaining_epochs": max(0, int(args.epochs) - int(start_epoch)),
            "resume_checkpoint": str(args.resume) if args.resume is not None else None,
            "best_val_loss_at_start": best_val_loss,
            "device_requested": args.device,
            "device_resolved": resolved_device,
            "seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "checkpoint_every": int(args.checkpoint_every),
            "optimizer": _optimizer_overview(optimizer),
            "scheduler": None if scheduler is None else scheduler.__class__.__name__,
        },
        "data": {
            "resolved_data_dir": str(data_dir),
            "num_loaded_events": len(event_sources),
            "source_files": sorted(
                set(source_files),
                key=lambda name: root_file_sort_key(Path(name)),
            ),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "class_counts": _jsonable(class_counts_by_split),
            "target_order_counts": _jsonable(target_order_counts_by_split),
            "cache": (
                _jsonable(event_sources.cache_info())
                if hasattr(event_sources, "cache_info")
                else None
            ),
        },
        "preprocessing": {
            "target_mode": args.target_mode,
            "valid_labels": list(args.valid_labels),
            "normalize_features": not bool(args.no_normalize_features),
            "feature_norm": _feature_norm_overview(feature_norm),
            "view_fn": getattr(view_fn, "__name__", str(view_fn)),
            "training_view_fn": (
                "precomputed"
                if training_view_fn is None
                else getattr(training_view_fn, "__name__", str(training_view_fn))
            ),
            "model_view_cache": _jsonable(view_cache_info),
        },
        "hyperparameters": _jsonable(vars(args)),
    }
    return _jsonable(overview)


def render_run_overview_markdown(overview):
    """Render a compact Markdown companion to ``run_overview.json``."""
    model = overview["model"]
    training = overview["training"]
    data = overview["data"]
    preprocessing = overview["preprocessing"]
    optimizer = training["optimizer"]

    lines = [
        "# Training Run Overview",
        "",
        "## Run",
        f"- Output directory: `{overview['run']['output_dir']}`",
        f"- Created UTC: `{overview['run']['created_at_utc']}`",
        f"- Script: `{overview['run']['script']}`",
        f"- Python: `{overview['run']['python']}`",
        f"- Platform: `{overview['run']['platform']}`",
        f"- Torch: `{overview['run']['torch_version']}`",
        "",
        "## Model",
        f"- Requested model: `{model['requested_name']}`",
        f"- Class: `{model['module']}.{model['class_name']}`",
        f"- Input dimension: `{model['input_dim']}`",
        f"- Output dimension: `{model['output_dim']}`",
        f"- Trainable parameters: `{model['trainable_parameters']}`",
        f"- Total parameters: `{model['total_parameters']}`",
        "",
        "### Constructor kwargs",
        "```json",
        json.dumps(model["constructor_kwargs"], indent=2, sort_keys=True),
        "```",
        "",
        "### Architecture",
        "```text",
        model["architecture"],
        "```",
        "",
        "## Training",
        f"- Start epoch: `{training['start_epoch']}`",
        f"- Requested epochs: `{training['requested_epochs']}`",
        f"- Remaining epochs: `{training['remaining_epochs']}`",
        f"- Device requested: `{training['device_requested']}`",
        f"- Device resolved: `{training['device_resolved']}`",
        f"- Seed: `{training['seed']}`",
        f"- Batch size: `{training['batch_size']}`",
        f"- Learning rate: `{training['learning_rate']}`",
        f"- Weight decay: `{training['weight_decay']}`",
        f"- Gradient clip: `{training['grad_clip']}`",
        f"- Checkpoint every: `{training['checkpoint_every']}`",
        f"- Resume checkpoint: `{training['resume_checkpoint']}`",
        "",
        "### Optimizer",
        f"- Class: `{optimizer['module']}.{optimizer['class_name']}`",
        "```json",
        json.dumps(optimizer["param_groups"], indent=2, sort_keys=True),
        "```",
        "",
        "## Data",
        f"- Resolved data dir: `{data['resolved_data_dir']}`",
        f"- Loaded events: `{data['num_loaded_events']}`",
        f"- Split sizes: `{json.dumps(data['split_sizes'], sort_keys=True)}`",
        f"- Shard cache: `{json.dumps(data['cache'], sort_keys=True)}`",
        "",
        "### Class counts",
        "```json",
        json.dumps(data["class_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "### Target order counts",
        "```json",
        json.dumps(data["target_order_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Preprocessing",
        f"- Target mode: `{preprocessing['target_mode']}`",
        f"- Valid labels: `{preprocessing['valid_labels']}`",
        f"- Normalize features: `{preprocessing['normalize_features']}`",
        f"- View function: `{preprocessing['view_fn']}`",
        f"- Training view function: `{preprocessing['training_view_fn']}`",
        f"- Model-view cache: `{json.dumps(preprocessing['model_view_cache'], sort_keys=True)}`",
        "",
        "### Feature normalization",
        "```json",
        json.dumps(preprocessing["feature_norm"], indent=2, sort_keys=True),
        "```",
        "",
        "## Hyperparameters",
        "| Name | Value |",
        "| --- | --- |",
    ]
    for key, value in overview["hyperparameters"].items():
        lines.append(f"| `{key}` | `{_format_value(value)}` |")

    lines.extend(
        [
            "",
            "## Parameter Tensors",
            "| Name | Shape | Numel | Trainable | Dtype |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for parameter in model["parameter_tensors"]:
        lines.append(
            "| `{name}` | `{shape}` | `{numel}` | `{trainable}` | `{dtype}` |".format(
                name=parameter["name"],
                shape=parameter["shape"],
                numel=parameter["numel"],
                trainable=parameter["trainable"],
                dtype=parameter["dtype"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def save_run_overview(run_dir, overview):
    """Save machine-readable and human-readable run overview artifacts."""
    run_dir = Path(run_dir)
    json_path = run_dir / "run_overview.json"
    markdown_path = run_dir / "run_overview.md"
    architecture_path = run_dir / "model_architecture.txt"
    save_json(json_path, overview)
    markdown_path.write_text(render_run_overview_markdown(overview), encoding="utf-8")
    architecture_path.write_text(overview["model"]["architecture"] + "\n", encoding="utf-8")
    return {
        "json": json_path,
        "markdown": markdown_path,
        "architecture": architecture_path,
    }


def log_run_overview(logger, overview, artifact_paths):
    """Log the key run overview details at startup."""
    model = overview["model"]
    training = overview["training"]
    data = overview["data"]
    hyperparameter_lines = "\n".join(
        f"  {key}: {_format_value(value)}"
        for key, value in overview["hyperparameters"].items()
    )
    logger.info(
        "Run overview artifacts: %s, %s, %s",
        artifact_paths["json"],
        artifact_paths["markdown"],
        artifact_paths["architecture"],
    )
    logger.info(
        "Model overview: %s.%s, input_dim=%s, output_dim=%s, trainable_parameters=%s",
        model["module"],
        model["class_name"],
        model["input_dim"],
        model["output_dim"],
        model["trainable_parameters"],
    )
    logger.info("Model architecture:\n%s", model["architecture"])
    logger.info(
        "Training overview: epochs=%s, start_epoch=%s, batch_size=%s, lr=%s, "
        "weight_decay=%s, grad_clip=%s, device=%s",
        training["requested_epochs"],
        training["start_epoch"],
        training["batch_size"],
        training["learning_rate"],
        training["weight_decay"],
        training["grad_clip"],
        training["device_resolved"],
    )
    logger.info(
        "Data overview: loaded_events=%s, split_sizes=%s, data_dir=%s",
        data["num_loaded_events"],
        data["split_sizes"],
        data["resolved_data_dir"],
    )
    logger.info("Data cache overview: %s", data["cache"])
    logger.info("Model-view cache overview: %s", overview["preprocessing"]["model_view_cache"])
    logger.info("All run hyperparameters:\n%s", hyperparameter_lines)
