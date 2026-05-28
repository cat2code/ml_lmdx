"""Validate all maintained models from one canonical combined tensor event."""

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from mldmx.datasets.ecal_tpad_loading import (
    apply_variable_count_target_mode,
    filter_noise_tensor_event,
    load_multi_sharded_tensor_events,
    load_processed_or_grouped_root_tensor_events,
)
from mldmx.datasets.model_views import (
    ecal_gravnet_view,
    ecal_tpad_gravnet_view,
    ecal_tpad_slot_model_view,
    ecal_tpad_transformer_view,
    ecal_transformer_view,
)
from mldmx.models import (
    ECalGravNet,
    ECalTpadGravNet,
    ECalTpadSlotModel,
    ECalTpadTransformer,
    ECalTransformer,
)
from mldmx.train.ecal_tpad_slot_model import compute_event_losses as compute_slot_event_losses
from mldmx.train.paths import resolve_existing_path


VALID_LABELS = (1, 2, 3)
TARGET_MODE = "canonical-y"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data/processed/ecal_tpad_3class_smoke"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data/ldmx_overlay_events_700k"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate maintained model forwards and backwards from one canonical event."
    )
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--processed-cache-root", type=Path, default=None)
    parser.add_argument(
        "--processed-source",
        action="append",
        nargs=3,
        metavar=("ELECTRON_COUNT", "LABEL", "CACHE_DIR"),
        help="Add one existing sharded cache source.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--events-per-source", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


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


def load_validation_event(args):
    if args.processed_cache_root is not None or args.processed_source is not None:
        logger = logging.getLogger("model-family-validation")
        events, _sources, selected_source, _root_files = load_multi_sharded_tensor_events(
            processed_sources_from_args(args),
            events_per_source=args.events_per_source,
            logger=logger,
        )
        if args.event_index >= len(events):
            raise ValueError(
                f"Requested event index {args.event_index}, but only loaded {len(events)} event(s)."
            )
        return events[args.event_index], Path("multi-source-sharded-cache"), False

    processed_dir = resolve_existing_path(args.processed_dir, project_root=PROJECT_ROOT)
    data_root = resolve_existing_path(args.data_root, project_root=PROJECT_ROOT)
    root_specs = [
        (2, "2e", data_root / "2e/events"),
        (3, "3e", data_root / "3e/events"),
    ]
    logger = logging.getLogger("model-family-validation")
    events, _sources, selected_source, root_files = load_processed_or_grouped_root_tensor_events(
        processed_dir=processed_dir,
        root_specs=root_specs,
        root_data_dir=data_root,
        events_per_source=max(1, args.event_index + 1),
        max_processed_events=args.event_index + 1,
        valid_labels=VALID_LABELS,
        filter_noise=True,
        allow_fewer_events=True,
        logger=logger,
        disable_progress=True,
        read_step_size=50,
    )
    if args.event_index >= len(events):
        raise ValueError(
            f"Requested event index {args.event_index}, but only loaded {len(events)} event(s)."
        )
    return events[args.event_index], Path(selected_source), bool(root_files)


def processed_noise_filter_setting(processed_dir):
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("filter_noise")


def report_noise_handling(source_path, loaded_from_root, raw_event):
    if "is_noise_target" in raw_event:
        count = int(raw_event["is_noise_target"].sum().item())
        print(f"noise_hits: stored={count}; filtered before common five-model validation")
        return
    filter_noise = True if loaded_from_root else processed_noise_filter_setting(source_path)
    if filter_noise is True:
        print("noise_hits: excluded (filter_noise=true); noise/background training is not validated")
    elif filter_noise is False:
        print("noise_hits: permitted (filter_noise=false; per-hit noise flags are not stored here)")
    else:
        print("noise_hits: unknown (processed manifest does not state filter_noise)")


def validate_provenance(event, original_physical_y):
    if "canonical_y" not in event:
        raise AssertionError("canonical-y target preparation did not expose event['canonical_y'].")
    if "origin_id_y" not in event:
        raise AssertionError("canonical-y target preparation did not retain event['origin_id_y'].")
    if not torch.equal(event["origin_id_y"], original_physical_y):
        raise AssertionError("event['origin_id_y'] does not preserve the pre-canonical physical origins.")


def run_baseline(name, model, view, device):
    model = model.to(device)
    model.train()
    logits = model(view["x"].to(device=device, dtype=torch.float32))
    supervised_logits = logits[view["ecal_mask"].to(device=device)]
    target = view["y"].to(device=device, dtype=torch.long)
    if supervised_logits.shape != (target.shape[0], len(VALID_LABELS)):
        raise AssertionError(
            f"{name}: expected supervised logits {(target.shape[0], len(VALID_LABELS))}, "
            f"got {tuple(supervised_logits.shape)}."
        )
    loss = F.cross_entropy(supervised_logits, target)
    if not bool(torch.isfinite(loss).item()):
        raise AssertionError(f"{name}: classification loss is not finite.")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError(f"{name}: backward produced no parameter gradients.")
    return tuple(view["x"].shape), tuple(supervised_logits.shape), float(loss.detach().item())


def run_slot_model(event, device):
    view = ecal_tpad_slot_model_view(event)
    model = ECalTpadSlotModel(
        in_dim=int(view["x"].shape[1]),
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        max_electrons=len(VALID_LABELS),
        dropout=0.0,
        use_type_embedding=True,
    ).to(device)
    loss_args = SimpleNamespace(
        lambda_origin=1.0,
        lambda_fraction=1.0,
        lambda_slot=0.5,
        lambda_count=1.0,
        origin_class_weights=None,
        count_class_weights=None,
    )
    losses = compute_slot_event_losses(model, view, device, loss_args)
    for key in ("total_loss", "origin_loss", "fraction_loss", "slot_loss", "count_loss"):
        if not bool(torch.isfinite(losses[key]).item()):
            raise AssertionError(f"ECalTpadSlotModel: {key} is not finite.")
    losses["total_loss"].backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("ECalTpadSlotModel: backward produced no parameter gradients.")
    supervised_shape = (int(losses["num_hits"]), model.num_classes)
    return tuple(view["x"].shape), supervised_shape, float(losses["total_loss"].detach().item())


def main():
    args = parse_args()
    if args.event_index < 0:
        raise ValueError("--event-index must be non-negative.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    raw_event, source_path, loaded_from_root = load_validation_event(args)
    event = filter_noise_tensor_event(raw_event)
    original_physical_y = event["physical_y"].clone()
    apply_variable_count_target_mode(
        event,
        valid_labels=VALID_LABELS,
        target_mode=TARGET_MODE,
        max_electrons=len(VALID_LABELS),
    )
    validate_provenance(event, original_physical_y)

    print(f"source: {source_path}")
    print(f"device: {device}")
    report_noise_handling(source_path, loaded_from_root, raw_event)
    print(
        f"provenance: present origin_id_y={sorted(set(event['origin_id_y'].tolist()))} "
        f"canonical_order={event['target_label_order']}"
    )

    baseline_checks = [
        (
            "ECalGravNet",
            ECalGravNet(
                in_dim=4,
                hidden_dim=32,
                out_dim=3,
                num_layers=1,
                space_dimensions=4,
                propagate_dimensions=16,
                k=8,
                dropout=0.0,
            ),
            ecal_gravnet_view(event),
        ),
        (
            "ECalTpadGravNet",
            ECalTpadGravNet(
                in_dim=8,
                hidden_dim=32,
                out_dim=3,
                num_layers=1,
                space_dimensions=4,
                propagate_dimensions=16,
                k=8,
                dropout=0.0,
            ),
            ecal_tpad_gravnet_view(event),
        ),
        (
            "ECalTransformer",
            ECalTransformer(
                in_dim=4,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                dropout=0.0,
                out_dim=3,
            ),
            ecal_transformer_view(event),
        ),
        (
            "ECalTpadTransformer",
            ECalTpadTransformer(
                in_dim=8,
                d_model=32,
                nhead=4,
                num_layers=1,
                dim_feedforward=64,
                dropout=0.0,
                out_dim=3,
            ),
            ecal_tpad_transformer_view(event),
        ),
    ]
    results = [
        (name, *run_baseline(name, model, view, device))
        for name, model, view in baseline_checks
    ]
    results.append(("ECalTpadSlotModel", *run_slot_model(event, device)))

    print(f"{'model':<21} {'input':<12} {'supervised output':<19} {'target mode':<12} {'loss':<10} status")
    for name, input_shape, output_shape, loss in results:
        print(
            f"{name:<21} {str(input_shape):<12} {str(output_shape):<19} "
            f"{TARGET_MODE:<12} {loss:<10.6f} pass"
        )


if __name__ == "__main__":
    main()
