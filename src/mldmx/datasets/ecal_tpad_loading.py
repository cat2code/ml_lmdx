import logging
from pathlib import Path

import torch

from mldmx.datasets.tensorize import (
    origin_energy_fraction_targets,
    tensorize_ecal_with_triggerpad_context,
)
from mldmx.io.root_reader import iter_ecal_rechits_with_truth_and_triggerpad_context


def canonical_axis_from_target_mode(target_mode):
    if target_mode == "canonical-x":
        return 0
    if target_mode == "canonical-y":
        return 1
    if target_mode == "canonical-z":
        return 2
    return None


def apply_target_mode(tensors, valid_labels, target_mode):
    axis = canonical_axis_from_target_mode(target_mode)
    if axis is None:
        tensors["target_class_names"] = [f"origin {label}" for label in valid_labels]
        tensors["target_label_order"] = list(valid_labels)
        return tensors

    original_physical_y = tensors["physical_y"].clone()
    original_fraction_target = tensors["fraction_target"].clone()
    pos = tensors["ecal_pos"]
    label_means = []
    for label in valid_labels:
        mask = original_physical_y == label
        if not bool(mask.any()):
            raise ValueError(
                f"Cannot canonicalize event {tensors['event_idx']}: origin label {label} is absent."
            )
        label_means.append((label, float(pos[mask, axis].mean().item())))

    ordered_labels = [label for label, _mean in sorted(label_means, key=lambda item: (item[1], item[0]))]
    label_to_canonical_class = {label: idx for idx, label in enumerate(ordered_labels)}
    original_label_to_column = {label: idx for idx, label in enumerate(valid_labels)}
    canonical_columns = [original_label_to_column[label] for label in ordered_labels]

    tensors["origin_id_y"] = original_physical_y
    tensors["origin_id_fraction_target"] = original_fraction_target
    tensors["y"] = torch.tensor(
        [label_to_canonical_class[int(label)] for label in original_physical_y.tolist()],
        dtype=torch.long,
    )
    tensors["physical_y"] = tensors["y"] + 1
    tensors["fraction_target"] = original_fraction_target[:, canonical_columns]
    tensors["target_label_order"] = ordered_labels
    axis_name = {0: "x", 1: "y", 2: "z"}[axis]
    tensors["target_class_names"] = [
        f"class {rank + 1}: {axis_name}-rank {rank + 1}"
        for rank, _label in enumerate(ordered_labels)
    ]
    return tensors


def ecal_tpad_event_to_tensors(
    event,
    event_idx,
    valid_labels,
    target_mode="physical-origin",
    filter_noise=True,
):
    tensors = tensorize_ecal_with_triggerpad_context(
        event,
        valid_labels=valid_labels,
        filter_noise=filter_noise,
    )
    tensors["event_idx"] = event_idx
    tensors["fraction_target"] = origin_energy_fraction_targets(
        event,
        keep_indices=tensors["keep_indices"],
        valid_labels=valid_labels,
    )
    return apply_target_mode(
        tensors,
        valid_labels=valid_labels,
        target_mode=target_mode,
    )


def _identity_progress(iterable, total=None, desc="", disable=False):
    return iterable


def _iter_root_events(root_files, max_events, read_step_size, logger):
    loaded = 0
    for root_file in root_files:
        if max_events is not None and loaded >= max_events:
            break

        remaining = None if max_events is None else max_events - loaded
        if logger is not None:
            requested = "all remaining" if remaining is None else str(remaining)
            logger.info("Reading %s event(s) from %s", requested, Path(root_file).name)

        for local_entry, event in iter_ecal_rechits_with_truth_and_triggerpad_context(
            root_file,
            max_events=remaining,
            step_size=read_step_size,
        ):
            yield Path(root_file), local_entry, event
            loaded += 1
            if max_events is not None and loaded >= max_events:
                return


def load_ecal_tpad_tensor_events(
    root_files,
    max_events,
    valid_labels,
    target_mode="physical-origin",
    filter_noise=True,
    allow_fewer_events=False,
    data_dir=None,
    logger=None,
    progress_factory=None,
    disable_progress=False,
    event_log_every=0,
    read_step_size=500,
):
    """
    Load labelled ECal + TriggerPadTracks ROOT events and tensorize them.

    The ROOT-specific work stays in mldmx.io.root_reader; this function owns the
    reusable dataset-level policy: walk files, assign global event indices,
    tensorize events, and record provenance for configs/manifests.
    """

    logger = logger or logging.getLogger(__name__)
    progress_factory = progress_factory or _identity_progress
    root_files = [Path(path) for path in root_files]
    valid_labels = tuple(valid_labels)

    selected_events = []
    event_sources = []
    progress_total = max_events if max_events is not None else None
    raw_event_iter = _iter_root_events(
        root_files,
        max_events=max_events,
        read_step_size=read_step_size,
        logger=logger,
    )

    logger.info(
        "Reading up to %s events from %s ROOT files",
        "all available" if max_events is None else max_events,
        len(root_files),
    )
    logger.info(
        "ROOT read chunk size: %s",
        "one read per file" if read_step_size is None else read_step_size,
    )

    progress = progress_factory(
        raw_event_iter,
        total=progress_total,
        desc="load ROOT events",
        disable=disable_progress,
    )
    for root_file, local_entry, event in progress:
        global_event_idx = len(selected_events)
        tensor_event = ecal_tpad_event_to_tensors(
            event,
            event_idx=global_event_idx,
            valid_labels=valid_labels,
            target_mode=target_mode,
            filter_noise=filter_noise,
        )
        selected_events.append(tensor_event)
        event_sources.append(
            {
                "event_idx": global_event_idx,
                "file": root_file.name,
                "entry": int(local_entry),
            }
        )

        if event_log_every > 0 and global_event_idx % event_log_every == 0:
            logger.info(
                "event=%s file=%s entry=%s selected_ecal_hits=%s tpad_nodes=%s labels=%s",
                global_event_idx,
                root_file.name,
                local_entry,
                int(tensor_event["ecal_mask"].sum().item()),
                int(tensor_event["tpad_mask"].sum().item()),
                sorted(set(tensor_event["physical_y"].tolist())),
            )

    if max_events is not None and len(selected_events) < max_events and not allow_fewer_events:
        location = f" in {data_dir}" if data_dir is not None else ""
        raise ValueError(
            f"Requested {max_events} events, but only found {len(selected_events)}{location}. "
            "Pass --allow-fewer-events to continue anyway."
        )

    logger.info("Loaded %s tensorized events", len(selected_events))
    return selected_events, event_sources
