"""Model-facing views derived from one canonical ECal + TriggerPadTracks event."""

import torch


COMBINED_FEATURE_DIM = 8
ECAL_FEATURE_SLICE = slice(2, 6)
_REQUIRED_FIELDS = ("x", "ecal_mask", "tpad_mask", "ecal_pos", "tpad", "y", "physical_y")
_METADATA_FIELDS = (
    "event_idx",
    "electron_count",
    "source_file",
    "source_entry",
    "source_label",
    "target_label_order",
    "origin_id_fraction_label_order",
    "target_class_names",
)
_OPTIONAL_TARGET_FIELDS = ("canonical_y", "fraction_target", "origin_id_fraction_target")
_OPTIONAL_HIT_FIELDS = ("ecal_input_energy", "ecal_raw_energy")
_OPTIONAL_TPAD_FIELDS = ("tpad_raw_pe",)


def validate_canonical_combined_event(event: dict) -> None:
    """Validate the existing combined tensor-event layout required by model views."""
    missing = [key for key in _REQUIRED_FIELDS if key not in event]
    if missing:
        raise KeyError(f"Canonical combined event is missing required field(s): {missing}.")

    x = event["x"]
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != COMBINED_FEATURE_DIM:
        shape = tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
        raise ValueError(
            f"Expected event['x'] to have shape [N, {COMBINED_FEATURE_DIM}], got {shape}."
        )
    num_nodes = x.shape[0]

    for key in ("ecal_mask", "tpad_mask"):
        mask = event[key]
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (num_nodes,):
            shape = tuple(mask.shape) if isinstance(mask, torch.Tensor) else type(mask).__name__
            raise ValueError(f"Expected event['{key}'] to be bool [{num_nodes}], got {shape}.")

    ecal_mask = event["ecal_mask"]
    tpad_mask = event["tpad_mask"]
    if bool((ecal_mask & tpad_mask).any().item()):
        raise ValueError("Canonical event ECal and TriggerPadTracks masks overlap.")
    if not bool((ecal_mask | tpad_mask).all().item()):
        raise ValueError("Canonical event contains nodes in neither detector mask.")

    num_ecal = int(ecal_mask.sum().item())
    num_tpad = int(tpad_mask.sum().item())
    _require_matrix(event, "ecal_pos", num_ecal, 3)
    _require_matrix(event, "tpad", num_tpad, 2)
    _require_hit_vector(event, "y", num_ecal)
    _require_hit_vector(event, "physical_y", num_ecal)

    for key in ("origin_id_y", "canonical_y"):
        if key in event:
            _require_hit_vector(event, key, num_ecal)
    for key in _OPTIONAL_HIT_FIELDS:
        if key in event:
            _require_hit_vector(event, key, num_ecal)
    for key in _OPTIONAL_TPAD_FIELDS:
        if key in event:
            value = event[key]
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or value.shape[0] != num_tpad
            ):
                shape = (
                    tuple(value.shape)
                    if isinstance(value, torch.Tensor)
                    else type(value).__name__
                )
                raise ValueError(
                    f"Expected event['{key}'] to align with {num_tpad} TriggerPadTracks, "
                    f"got {shape}."
                )
    for key in ("fraction_target", "origin_id_fraction_target"):
        if key in event:
            target = event[key]
            if not isinstance(target, torch.Tensor) or target.ndim != 2 or target.shape[0] != num_ecal:
                shape = tuple(target.shape) if isinstance(target, torch.Tensor) else type(target).__name__
                raise ValueError(
                    f"Expected event['{key}'] to have first dimension {num_ecal}, got {shape}."
                )


def _require_matrix(event: dict, key: str, rows: int, columns: int) -> None:
    value = event[key]
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape != (rows, columns):
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise ValueError(f"Expected event['{key}'] to have shape [{rows}, {columns}], got {shape}.")


def _require_hit_vector(event: dict, key: str, num_ecal: int) -> None:
    value = event[key]
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != num_ecal:
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise ValueError(f"Expected event['{key}'] to align with {num_ecal} ECal hits, got {shape}.")


def _hit_targets_and_metadata(event: dict) -> dict:
    physical_origin = event.get("origin_id_y", event["physical_y"])
    target = event.get("canonical_y", event["y"])
    fields = {
        "y": target,
        "origin_id_y": physical_origin,
    }
    for key in _OPTIONAL_TARGET_FIELDS:
        if key in event:
            fields[key] = event[key]
    for key in _OPTIONAL_HIT_FIELDS:
        if key in event:
            fields[key] = event[key]
    for key in _METADATA_FIELDS:
        if key in event:
            fields[key] = event[key]
    return fields


def _ecal_only_view(event: dict) -> dict:
    validate_canonical_combined_event(event)
    num_ecal = int(event["ecal_mask"].sum().item())
    view = {
        "x": event["x"][event["ecal_mask"], ECAL_FEATURE_SLICE],
        "ecal_mask": torch.ones((num_ecal,), dtype=torch.bool, device=event["ecal_mask"].device),
        "ecal_pos": event["ecal_pos"],
    }
    view.update(_hit_targets_and_metadata(event))
    return view


def _combined_view(event: dict) -> dict:
    validate_canonical_combined_event(event)
    view = {
        "x": event["x"],
        "ecal_mask": event["ecal_mask"],
        "tpad_mask": event["tpad_mask"],
        "ecal_pos": event["ecal_pos"],
        "tpad": event["tpad"],
    }
    for key in _OPTIONAL_TPAD_FIELDS:
        if key in event:
            view[key] = event[key]
    view.update(_hit_targets_and_metadata(event))
    return view


def ecal_transformer_view(event: dict) -> dict:
    """Return physical ECal hit-token features for ``ECalTransformer``."""
    return _ecal_only_view(event)


def ecal_tpad_transformer_view(event: dict) -> dict:
    """Return full mixed-detector tokens for ``ECalTpadTransformer``."""
    return _combined_view(event)


def ecal_gravnet_view(event: dict) -> dict:
    """Return physical ECal node features for ``ECalGravNet``."""
    return _ecal_only_view(event)


def ecal_tpad_gravnet_view(event: dict) -> dict:
    """Return full mixed-detector nodes for ``ECalTpadGravNet``."""
    return _combined_view(event)


def ecal_tpad_slot_model_view(event: dict) -> dict:
    """Validate and return the unchanged full input for ``ECalTpadSlotModel``."""
    validate_canonical_combined_event(event)
    return event
