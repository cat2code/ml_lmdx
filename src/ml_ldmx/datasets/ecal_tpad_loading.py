import logging
from pathlib import Path

import torch

from ml_ldmx.datasets.tensorize import (
    ECAL_ENERGY_TRANSFORMS,
    TPAD_PE_TRANSFORMS,
    origin_energy_fraction_targets,
    tensorize_ecal_with_triggerpad_context,
)
from ml_ldmx.datasets.ecal_tpad_dataset import ECalTriggerPadTensorDataset
from ml_ldmx.datasets.ecal_tpad_shards import (
    MultiShardedECalTpadDataset,
    ShardedECalTpadDataset,
    has_sharded_tensor_cache,
    prepare_sharded_tensor_cache,
    validate_sharded_cache_request,
    validate_sharded_tensor_cache,
)
from ml_ldmx.io.root_files import find_root_files
from ml_ldmx.io.root_reader import iter_ecal_rechits_with_truth_and_triggerpad_context


def canonical_axis_from_target_mode(target_mode):
    if target_mode == "canonical-x":
        return 0
    if target_mode == "canonical-y":
        return 1
    if target_mode == "canonical-z":
        return 2
    return None


def _fraction_label_order(tensors, fraction_target, fallback_labels):
    """Resolve the physical-origin label represented by each fraction column."""
    for key in ("origin_id_fraction_label_order", "target_label_order"):
        value = tensors.get(key)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1).tolist()
        labels = [int(label) for label in value]
        if len(labels) == fraction_target.shape[1]:
            return labels

    fallback_labels = [int(label) for label in fallback_labels]
    if len(fallback_labels) == fraction_target.shape[1]:
        return fallback_labels
    raise ValueError(
        "Cannot map physical-origin fraction columns to labels: "
        f"found {fraction_target.shape[1]} columns without a matching stored label order."
    )


def apply_target_mode(tensors, valid_labels, target_mode):
    axis = canonical_axis_from_target_mode(target_mode)
    if axis is None:
        tensors["target_class_names"] = [f"origin {label}" for label in valid_labels]
        tensors["target_label_order"] = list(valid_labels)
        return tensors

    original_physical_y = tensors["physical_y"].clone()
    original_fraction_target = tensors.get(
        "origin_id_fraction_target", tensors["fraction_target"]
    ).clone()
    fraction_label_order = _fraction_label_order(
        tensors,
        original_fraction_target,
        fallback_labels=valid_labels,
    )
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
    original_label_to_column = {
        label: idx for idx, label in enumerate(fraction_label_order)
    }
    canonical_columns = [original_label_to_column[label] for label in ordered_labels]

    tensors["origin_id_y"] = original_physical_y
    tensors["origin_id_fraction_target"] = original_fraction_target
    tensors["origin_id_fraction_label_order"] = fraction_label_order
    tensors["y"] = torch.tensor(
        [label_to_canonical_class[int(label)] for label in original_physical_y.tolist()],
        dtype=torch.long,
    )
    tensors["canonical_y"] = tensors["y"].clone()
    tensors["physical_y"] = tensors["y"] + 1
    tensors["fraction_target"] = original_fraction_target[:, canonical_columns]
    tensors["target_label_order"] = ordered_labels
    axis_name = {0: "x", 1: "y", 2: "z"}[axis]
    tensors["target_class_names"] = [
        f"class {rank + 1}: {axis_name}-rank {rank + 1}"
        for rank, _label in enumerate(ordered_labels)
    ]
    return tensors


def apply_variable_count_target_mode(tensors, valid_labels, target_mode, max_electrons):
    """
    Apply spatially ordered slot targets for events with varying multiplicity.

    This is the mixed-count counterpart to ``apply_target_mode``. Canonical
    targets retain their original physical-origin labels in ``origin_id_y`` and
    ``origin_id_fraction_target`` for provenance.
    """
    original_fraction = None
    fraction_label_order = None
    if "fraction_target" in tensors:
        original_fraction = tensors.get(
            "origin_id_fraction_target", tensors["fraction_target"]
        ).clone()
        fraction_label_order = _fraction_label_order(
            tensors,
            original_fraction,
            fallback_labels=valid_labels,
        )

    noise_mask = tensors.get("is_noise_target")
    if noise_mask is not None:
        noise_mask = noise_mask.to(dtype=torch.bool)
        if noise_mask.shape != tensors["physical_y"].shape:
            raise ValueError(
                "event['is_noise_target'] must align with event['physical_y'] for canonicalization."
            )
    else:
        noise_mask = torch.zeros_like(tensors["physical_y"], dtype=torch.bool)

    axis = canonical_axis_from_target_mode(target_mode)
    if axis is None:
        if bool(noise_mask.any().item()):
            raise ValueError("Explicit noise supervision currently requires a canonical target mode.")
        tensors["target_class_names"] = [f"origin {label}" for label in valid_labels]
        tensors["target_label_order"] = list(valid_labels)
        return tensors

    if "physical_y" not in tensors or "ecal_pos" not in tensors:
        raise KeyError("Canonical slot target mode requires event['physical_y'] and event['ecal_pos'].")

    assignment_y = tensors["physical_y"].clone()
    original_physical_y = tensors.get("origin_id_y", assignment_y).clone()
    pos = tensors["ecal_pos"]
    electron_mask = ~noise_mask
    present_labels = sorted({int(label) for label in assignment_y[electron_mask].tolist()})
    if len(present_labels) > max_electrons:
        raise ValueError(
            f"Event has {len(present_labels)} origin labels, but max_electrons={max_electrons}."
        )

    label_means = []
    for label in present_labels:
        mask = electron_mask & (assignment_y == label)
        label_means.append((label, float(pos[mask, axis].mean().item())))
    ordered_labels = [label for label, _mean in sorted(label_means, key=lambda item: (item[1], item[0]))]
    label_to_slot = {label: slot_idx + 1 for slot_idx, label in enumerate(ordered_labels)}

    tensors["origin_id_y"] = original_physical_y
    canonical_y = torch.full_like(assignment_y, -1)
    model_target = torch.zeros_like(assignment_y)
    for label, slot_class in label_to_slot.items():
        label_mask = electron_mask & (assignment_y == label)
        canonical_y[label_mask] = slot_class - 1
        model_target[label_mask] = slot_class
    tensors["y"] = canonical_y
    tensors["canonical_y"] = canonical_y.clone()
    tensors["physical_y"] = model_target
    tensors["target_label_order"] = ordered_labels

    if original_fraction is not None:
        tensors["origin_id_fraction_target"] = original_fraction
        tensors["origin_id_fraction_label_order"] = fraction_label_order
        physical_label_to_column = {
            label: idx for idx, label in enumerate(fraction_label_order)
        }
        canonical_fraction = torch.zeros(
            (original_fraction.shape[0], max_electrons),
            dtype=original_fraction.dtype,
            device=original_fraction.device,
        )
        for slot_idx, label in enumerate(ordered_labels):
            source_col = physical_label_to_column.get(int(label))
            if source_col is not None:
                canonical_fraction[electron_mask, slot_idx] = original_fraction[electron_mask, source_col]
        tensors["fraction_target"] = canonical_fraction

    axis_name = {0: "x", 1: "y", 2: "z"}[axis]
    electron_target_names = [
        f"slot {slot_idx + 1}: {axis_name}-rank {slot_idx + 1}"
        for slot_idx in range(max_electrons)
    ]
    tensors["target_class_names"] = (
        ["noise/background"] + electron_target_names
        if bool(noise_mask.any().item())
        else electron_target_names
    )
    return tensors


def apply_variable_count_target_mode_to_events(events, valid_labels, target_mode, max_electrons):
    for event in events:
        apply_variable_count_target_mode(
            event,
            valid_labels=valid_labels,
            target_mode=target_mode,
            max_electrons=max_electrons,
        )


def filter_noise_tensor_event(event):
    """Remove stored ECal noise rows while retaining the canonical combined layout."""
    noise_mask = event.get("is_noise_target")
    if noise_mask is None:
        return event
    noise_mask = noise_mask.to(dtype=torch.bool)
    if noise_mask.ndim != 1 or noise_mask.shape != event["physical_y"].shape:
        raise ValueError("event['is_noise_target'] must align with ECal hit targets.")
    if not bool(noise_mask.any().item()):
        return event

    filtered = dict(event)
    ecal_mask = event["ecal_mask"].to(dtype=torch.bool)
    tpad_mask = event["tpad_mask"].to(dtype=torch.bool)
    ecal_rows = event["x"][ecal_mask][~noise_mask]
    tpad_rows = event["x"][tpad_mask]
    filtered["x"] = torch.cat([ecal_rows, tpad_rows], dim=0)
    filtered["ecal_mask"] = torch.cat(
        [
            torch.ones((ecal_rows.shape[0],), dtype=torch.bool),
            torch.zeros((tpad_rows.shape[0],), dtype=torch.bool),
        ]
    )
    filtered["tpad_mask"] = ~filtered["ecal_mask"]

    for key in (
        "ecal_pos",
        "pos",
        "ecal_input_energy",
        "ecal_raw_energy",
        "y",
        "physical_y",
        "origin_id_y",
        "canonical_y",
        "keep_indices",
        "is_noise_target",
        "fraction_target",
        "origin_id_fraction_target",
    ):
        if key in event:
            filtered[key] = event[key][~noise_mask]
    return filtered


def ecal_tpad_event_to_tensors(
    event,
    event_idx,
    valid_labels,
    target_mode="physical-origin",
    filter_noise=True,
    supervise_noise=False,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
):
    tensors = tensorize_ecal_with_triggerpad_context(
        event,
        valid_labels=valid_labels,
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        ecal_energy_transform=ecal_energy_transform,
        tpad_pe_transform=tpad_pe_transform,
    )
    tensors["event_idx"] = event_idx
    tensors["fraction_target"] = origin_energy_fraction_targets(
        event,
        keep_indices=tensors["keep_indices"],
        valid_labels=valid_labels,
        is_noise_target=tensors.get("is_noise_target"),
    )
    return apply_target_mode(
        tensors,
        valid_labels=valid_labels,
        target_mode=target_mode,
    )


def has_processed_tensor_events(processed_dir):
    processed_dir = Path(processed_dir)
    return processed_dir.exists() and any(processed_dir.glob("event_*.pt"))


def load_processed_tensor_events(processed_dir, max_events=None, logger=None):
    """Load canonical combined tensor events without changing their stored format."""
    logger = logger or logging.getLogger(__name__)
    processed_dir = Path(processed_dir)
    dataset = ECalTriggerPadTensorDataset(processed_dir)
    limit = len(dataset) if max_events is None else min(max_events, len(dataset))
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
    return events, event_sources


def load_or_create_sharded_tensor_events(
    processed_cache,
    root_specs,
    valid_labels,
    max_events=None,
    filter_noise=False,
    supervise_noise=True,
    force=False,
    skip_existing=True,
    max_root_files=None,
    max_events_per_root_file=None,
    shard_cache_size=1,
    allow_incomplete_cache=False,
    logger=None,
    read_step_size=500,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
):
    """Reuse or create an ML-ready sharded cache and return its lazy dataset."""
    logger = logger or logging.getLogger(__name__)
    processed_cache = Path(processed_cache)
    build_cache = force or not has_sharded_tensor_cache(processed_cache)
    if not build_cache:
        validate_sharded_cache_request(
            processed_cache,
            root_specs=root_specs,
            valid_labels=valid_labels,
            filter_noise=filter_noise,
            supervise_noise=supervise_noise,
            max_root_files=max_root_files,
            max_events_per_root_file=max_events_per_root_file,
            ecal_energy_transform=ecal_energy_transform,
            tpad_pe_transform=tpad_pe_transform,
        )
        try:
            validate_sharded_tensor_cache(
                processed_cache,
                load_shards=False,
                allow_incomplete=allow_incomplete_cache,
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.info("Resuming incomplete sharded tensor cache (%s)", exc)
            build_cache = True
    if build_cache:
        prepare_sharded_tensor_cache(
            cache_dir=processed_cache,
            root_specs=root_specs,
            valid_labels=valid_labels,
            filter_noise=filter_noise,
            supervise_noise=supervise_noise,
            force=force,
            skip_existing=skip_existing,
            max_root_files=max_root_files,
            max_events_per_root_file=max_events_per_root_file,
            read_step_size=read_step_size,
            ecal_energy_transform=ecal_energy_transform,
            tpad_pe_transform=tpad_pe_transform,
            logger=logger,
        )
    elif allow_incomplete_cache:
        logger.warning("Using only valid completed shards from an incomplete cache: %s", processed_cache)
    else:
        logger.info("Using existing sharded tensor cache: %s", processed_cache)
    dataset = ShardedECalTpadDataset(
        processed_cache,
        max_events=max_events,
        shard_cache_size=shard_cache_size,
        allow_incomplete=allow_incomplete_cache,
    )
    manifest_spec = dataset.metadata.get("cache_spec", {})
    requested = {
        "valid_labels": list(valid_labels),
        "filter_noise": bool(filter_noise),
        "supervise_noise": bool(supervise_noise),
        "ecal_energy_transform": ecal_energy_transform,
        "tpad_pe_transform": tpad_pe_transform,
    }
    mismatches = {
        key: (manifest_spec.get(key), value)
        for key, value in requested.items()
        if manifest_spec.get(key, "raw" if key in ("ecal_energy_transform", "tpad_pe_transform") else None) != value
    }
    if mismatches:
        raise ValueError(
            f"Existing sharded cache target/filter settings do not match requested training: {mismatches}."
        )
    logger.info("Shard cache configuration: %s", dataset.cache_info())
    return dataset, dataset, processed_cache, dataset.root_files


def load_multi_sharded_tensor_events(
    processed_sources,
    max_events=None,
    events_per_source=None,
    shard_cache_size=1,
    allow_incomplete_cache=False,
    logger=None,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
):
    """Load multiple existing sharded caches as one lazy event dataset."""
    logger = logger or logging.getLogger(__name__)
    sources = []
    for electron_count, source_label, cache_dir in processed_sources:
        cache_dir = Path(cache_dir)
        dataset = ShardedECalTpadDataset(
            cache_dir,
            max_events=events_per_source,
            shard_cache_size=shard_cache_size,
            allow_incomplete=allow_incomplete_cache,
        )
        cache_spec = dataset.metadata.get("cache_spec", {})
        cache_transform = cache_spec.get("ecal_energy_transform", "raw")
        cache_tpad_transform = cache_spec.get("tpad_pe_transform", "raw")
        if cache_transform != ecal_energy_transform:
            raise ValueError(
                f"Sharded cache {cache_dir} stores ecal_energy_transform={cache_transform!r}, "
                f"but {ecal_energy_transform!r} was requested."
            )
        if cache_tpad_transform != tpad_pe_transform:
            raise ValueError(
                f"Sharded cache {cache_dir} stores tpad_pe_transform={cache_tpad_transform!r}, "
                f"but {tpad_pe_transform!r} was requested."
            )
        sources.append(
            {
                "electron_count": int(electron_count),
                "source_label": source_label,
                "cache_dir": cache_dir,
                "dataset": dataset,
            }
        )
        logger.info(
            "Using sharded source %s (%se): %s event(s) from %s",
            source_label,
            electron_count,
            len(dataset),
            cache_dir,
        )
    combined = MultiShardedECalTpadDataset(sources, max_events=max_events)
    logger.info(
        "Loaded lazy multi-source sharded dataset with %s total event(s): %s",
        len(combined),
        combined.source_summaries,
    )
    logger.info("Shard cache configuration: %s", combined.cache_info())
    return combined, combined, [source["cache_dir"] for source in sources], combined.root_files


def attach_root_source_metadata(event, source, global_event_idx, electron_count=None, source_label=None):
    """Attach selected ROOT source metadata to one canonical tensor event."""
    event["event_idx"] = torch.tensor(global_event_idx, dtype=torch.long)
    if electron_count is not None:
        event["electron_count"] = torch.tensor(electron_count, dtype=torch.long)
    if source_label is not None:
        event["source_label"] = source_label
    event["source_file"] = source["file"]
    event["source_entry"] = int(source["entry"])


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
    supervise_noise=False,
    allow_fewer_events=False,
    data_dir=None,
    logger=None,
    progress_factory=None,
    disable_progress=False,
    event_log_every=0,
    read_step_size=500,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
):
    """
    Load labelled ECal + TriggerPadTracks ROOT events and tensorize them.

    The ROOT-specific work stays in ml_ldmx.io.root_reader; this function owns the
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
            supervise_noise=supervise_noise,
            ecal_energy_transform=ecal_energy_transform,
            tpad_pe_transform=tpad_pe_transform,
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


def load_grouped_root_tensor_events(
    root_specs,
    events_per_source,
    valid_labels,
    filter_noise=True,
    supervise_noise=False,
    allow_fewer_events=False,
    logger=None,
    progress_factory=None,
    disable_progress=False,
    event_log_every=0,
    read_step_size=500,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
):
    """
    Load multiple labelled ROOT source groups into the canonical combined form.

    ``root_specs`` contains ``(electron_count, source_label, data_dir)`` tuples.
    Events are tensorized with physical-origin targets first so callers can
    apply one comparison target convention across mixed source groups.
    """
    logger = logger or logging.getLogger(__name__)
    events = []
    event_sources = []
    root_files_used = []

    for electron_count, source_label, data_dir in root_specs:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Could not find {source_label} ROOT directory: {data_dir}")
        root_files = find_root_files(data_dir)
        root_files_used.extend(root_files)
        logger.info(
            "Loading up to %s %s events from %s",
            events_per_source,
            source_label,
            data_dir,
        )
        loaded_events, sources = load_ecal_tpad_tensor_events(
            root_files=root_files,
            max_events=events_per_source,
            valid_labels=tuple(valid_labels),
            target_mode="physical-origin",
            filter_noise=filter_noise,
            supervise_noise=supervise_noise,
            allow_fewer_events=allow_fewer_events,
            data_dir=data_dir,
            logger=logger,
            progress_factory=progress_factory,
            disable_progress=disable_progress,
            event_log_every=event_log_every,
            read_step_size=read_step_size,
            ecal_energy_transform=ecal_energy_transform,
            tpad_pe_transform=tpad_pe_transform,
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
            event_sources.append(
                {
                    **source,
                    "event_idx": global_event_idx,
                    "electron_count": electron_count,
                    "source_label": source_label,
                    "source_dir": str(data_dir),
                }
            )
            events.append(event)

    return events, event_sources, root_files_used


def load_processed_or_grouped_root_tensor_events(
    processed_dir,
    root_specs,
    root_data_dir,
    events_per_source,
    max_processed_events,
    valid_labels,
    filter_noise=True,
    supervise_noise=False,
    allow_fewer_events=False,
    logger=None,
    progress_factory=None,
    disable_progress=False,
    event_log_every=0,
    read_step_size=500,
    shard_cache_size=1,
    allow_incomplete_sharded_cache=False,
    ecal_energy_transform="raw",
    tpad_pe_transform="raw",
):
    """
    Select an existing processed canonical dataset or build events from ROOT groups.

    Sharded processed caches are preferred when ``processed_dir`` contains both
    shard metadata and legacy ``event_*.pt`` files. This keeps local per-event
    smoke datasets working while making the scalable path the default for large
    processed directories.
    """
    logger = logger or logging.getLogger(__name__)
    if ecal_energy_transform not in ECAL_ENERGY_TRANSFORMS:
        raise ValueError(
            f"Unknown ECal energy transform {ecal_energy_transform!r}; "
            f"expected one of {ECAL_ENERGY_TRANSFORMS}."
        )
    if tpad_pe_transform not in TPAD_PE_TRANSFORMS:
        raise ValueError(
            f"Unknown TriggerPadTracks pe transform {tpad_pe_transform!r}; "
            f"expected one of {TPAD_PE_TRANSFORMS}."
        )
    processed_dir = Path(processed_dir)
    if has_sharded_tensor_cache(processed_dir):
        logger.info("Using sharded processed tensor dataset: %s", processed_dir)
        dataset = ShardedECalTpadDataset(
            processed_dir,
            max_events=max_processed_events,
            shard_cache_size=shard_cache_size,
            allow_incomplete=allow_incomplete_sharded_cache,
        )
        cache_spec = dataset.metadata.get("cache_spec", {})
        cache_transform = cache_spec.get("ecal_energy_transform", "raw")
        cache_tpad_transform = cache_spec.get("tpad_pe_transform", "raw")
        if cache_transform != ecal_energy_transform:
            raise ValueError(
                f"Processed sharded dataset {processed_dir} stores "
                f"ecal_energy_transform={cache_transform!r}, but {ecal_energy_transform!r} was requested."
            )
        if cache_tpad_transform != tpad_pe_transform:
            raise ValueError(
                f"Processed sharded dataset {processed_dir} stores "
                f"tpad_pe_transform={cache_tpad_transform!r}, but {tpad_pe_transform!r} was requested."
            )
        logger.info("Shard cache configuration: %s", dataset.cache_info())
        return dataset, dataset, processed_dir, dataset.root_files

    if has_processed_tensor_events(processed_dir):
        if supervise_noise:
            raise ValueError(
                "Explicit noise supervision requires ROOT-backed tensorization; "
                "existing processed events do not contain noise targets."
            )
        logger.info("Using processed tensor dataset: %s", processed_dir)
        metadata = ECalTriggerPadTensorDataset(processed_dir).metadata
        stored_transform = metadata.get(
            "ecal_energy_transform",
            metadata.get("cache_spec", {}).get("ecal_energy_transform", "raw"),
        )
        stored_tpad_transform = metadata.get(
            "tpad_pe_transform",
            metadata.get("cache_spec", {}).get("tpad_pe_transform", "raw"),
        )
        if stored_transform != ecal_energy_transform:
            raise ValueError(
                f"Processed tensor dataset {processed_dir} stores "
                f"ecal_energy_transform={stored_transform!r}, but {ecal_energy_transform!r} was requested."
            )
        if stored_tpad_transform != tpad_pe_transform:
            raise ValueError(
                f"Processed tensor dataset {processed_dir} stores "
                f"tpad_pe_transform={stored_tpad_transform!r}, but {tpad_pe_transform!r} was requested."
            )
        events, event_sources = load_processed_tensor_events(
            processed_dir,
            max_events=max_processed_events,
            logger=logger,
        )
        return events, event_sources, processed_dir, []

    root_data_dir = Path(root_data_dir)
    logger.info(
        "Processed dir %s not found or empty; loading grouped ROOT events from %s",
        processed_dir,
        root_data_dir,
    )
    events, event_sources, root_files = load_grouped_root_tensor_events(
        root_specs=root_specs,
        events_per_source=events_per_source,
        valid_labels=valid_labels,
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        allow_fewer_events=allow_fewer_events,
        logger=logger,
        progress_factory=progress_factory,
        disable_progress=disable_progress,
        event_log_every=event_log_every,
        read_step_size=read_step_size,
        ecal_energy_transform=ecal_energy_transform,
        tpad_pe_transform=tpad_pe_transform,
    )
    return events, event_sources, root_data_dir, root_files
