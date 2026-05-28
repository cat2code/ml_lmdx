"""Scalable sharded storage for canonical ECal + TriggerPad tensor events."""

from bisect import bisect_right
from collections import OrderedDict
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mldmx.io.root_files import find_root_files
from mldmx.io.root_reader import iter_ecal_rechits_with_truth_and_triggerpad_context


SHARD_CACHE_SCHEMA_VERSION = 1
SHARD_PAYLOAD_SCHEMA_VERSION = 1
FEATURE_LAYOUT = [
    "is_ecal",
    "is_tpad",
    "ecal_x",
    "ecal_y",
    "ecal_z",
    "ecal_energy",
    "tpad_centroid",
    "tpad_pe",
]


def _load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def _root_file_metadata(path, electron_count, source_label, source_dir):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "source_dir": str(Path(source_dir).resolve()),
        "source_label": source_label,
        "electron_count": int(electron_count) if electron_count is not None else None,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _root_sources(root_specs, max_root_files=None):
    sources = []
    for electron_count, source_label, source_dir in root_specs:
        source_dir = Path(source_dir)
        root_files = find_root_files(source_dir)
        if max_root_files is not None:
            root_files = root_files[:max_root_files]
        sources.extend(
            _root_file_metadata(path, electron_count, source_label, source_dir)
            for path in root_files
        )
    return sources


def _cache_spec(
    root_sources,
    valid_labels,
    filter_noise,
    supervise_noise,
    max_events_per_root_file,
):
    return {
        "reader": "ecal_tpad_sharded",
        "schema_version": SHARD_CACHE_SCHEMA_VERSION,
        "root_sources": root_sources,
        "valid_labels": list(valid_labels),
        "filter_noise": bool(filter_noise),
        "supervise_noise": bool(supervise_noise),
        "stored_target_mode": "physical-origin",
        "max_events_per_root_file": max_events_per_root_file,
        "feature_layout": FEATURE_LAYOUT,
    }


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def has_sharded_tensor_cache(cache_dir):
    cache_dir = Path(cache_dir)
    return (cache_dir / "manifest.json").exists() and (cache_dir / "index.json").exists()


def validate_sharded_cache_request(
    cache_dir,
    root_specs,
    valid_labels,
    filter_noise=True,
    supervise_noise=False,
    max_root_files=None,
    max_events_per_root_file=None,
):
    """Require an existing cache to correspond to the requested raw dataset/settings."""
    manifest = _load_json(Path(cache_dir) / "manifest.json")
    requested_spec = _cache_spec(
        root_sources=_root_sources(root_specs, max_root_files=max_root_files),
        valid_labels=tuple(valid_labels),
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        max_events_per_root_file=max_events_per_root_file,
    )
    if manifest.get("cache_spec") != requested_spec:
        raise ValueError(
            f"Existing sharded cache does not match requested ROOT inputs/settings: {cache_dir}. "
            "Choose a different --processed-cache or pass --force-sharded-cache."
        )
    return requested_spec


def _validate_shard_payload(payload, expected_source):
    if not isinstance(payload, dict) or payload.get("schema_version") != SHARD_PAYLOAD_SCHEMA_VERSION:
        return None
    if payload.get("source") != expected_source:
        return None
    events = payload.get("events")
    if not isinstance(events, list):
        return None
    return len(events)


def _index_payload(shard_entries, skipped_sources, total_events):
    return {
        "cache_schema_version": SHARD_CACHE_SCHEMA_VERSION,
        "num_events": total_events,
        "shards": shard_entries,
        "skipped_sources": skipped_sources,
    }


def _resume_prefix_from_index(index_path, root_sources, resume_from_root_index):
    """Trust already indexed sources before a requested 1-based resume position."""
    if resume_from_root_index <= 1:
        return [], [], 0
    if not index_path.exists():
        raise ValueError(
            f"--resume-from-root-index={resume_from_root_index} requires an existing index: {index_path}"
        )

    prior_index = _load_json(index_path)
    prefix_sources = root_sources[: resume_from_root_index - 1]
    prefix_paths = {source["path"] for source in prefix_sources}
    shard_entries = [
        entry for entry in prior_index.get("shards", [])
        if entry.get("source", {}).get("path") in prefix_paths
    ]
    skipped_sources = [
        entry for entry in prior_index.get("skipped_sources", [])
        if entry.get("source", {}).get("path") in prefix_paths
    ]
    indexed_by_path = {entry.get("source", {}).get("path"): entry.get("source") for entry in shard_entries}
    skipped_by_path = {entry.get("source", {}).get("path"): entry.get("source") for entry in skipped_sources}
    accounted_sources = [
        indexed_by_path.get(source["path"], skipped_by_path.get(source["path"]))
        for source in prefix_sources
    ]
    if accounted_sources != prefix_sources:
        raise ValueError(
            f"Cannot resume at ROOT index {resume_from_root_index}: existing index does not account "
            "for every earlier ROOT source."
        )

    total_events = 0
    for entry in shard_entries:
        if entry.get("event_start") != total_events:
            raise ValueError(f"Cannot resume from non-contiguous shard event offsets in {index_path}")
        total_events = entry.get("event_stop")
    return shard_entries, skipped_sources, int(total_events)


def prepare_sharded_tensor_cache(
    cache_dir,
    root_specs,
    valid_labels,
    filter_noise=True,
    supervise_noise=False,
    force=False,
    skip_existing=True,
    max_root_files=None,
    max_events_per_root_file=None,
    read_step_size=500,
    skip_failed_root_files=False,
    resume_from_root_index=1,
    logger=None,
):
    """Create or resume a one-ROOT-file-per-shard canonical tensor cache."""
    from mldmx.datasets.ecal_tpad_loading import (
        attach_root_source_metadata,
        ecal_tpad_event_to_tensors,
    )

    logger = logger or logging.getLogger(__name__)
    if resume_from_root_index < 1:
        raise ValueError("resume_from_root_index must be at least 1.")
    cache_dir = Path(cache_dir)
    shards_dir = cache_dir / "shards"
    root_sources = _root_sources(root_specs, max_root_files=max_root_files)
    spec = _cache_spec(
        root_sources=root_sources,
        valid_labels=tuple(valid_labels),
        filter_noise=filter_noise,
        supervise_noise=supervise_noise,
        max_events_per_root_file=max_events_per_root_file,
    )
    manifest_path = cache_dir / "manifest.json"
    index_path = cache_dir / "index.json"

    if manifest_path.exists() and not force:
        manifest = _load_json(manifest_path)
        if manifest.get("cache_spec") != spec:
            raise ValueError(
                f"Existing sharded cache metadata does not match requested ROOT inputs/settings: {cache_dir}. "
                "Choose a different --processed-cache or pass --force-sharded-cache."
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "cache_schema_version": SHARD_CACHE_SCHEMA_VERSION,
        "format": "ecal_tpad_root_file_shards",
        "cache_spec": spec,
        "feature_layout": FEATURE_LAYOUT,
        "filter_noise": bool(filter_noise),
        "supervise_noise": bool(supervise_noise),
        "valid_labels": list(valid_labels),
    }
    _write_json(manifest_path, manifest)

    prior_skipped_by_path = {}
    if skip_failed_root_files and index_path.exists() and not force:
        try:
            prior_index = _load_json(index_path)
            prior_skipped_by_path = {
                entry["source"]["path"]: entry
                for entry in prior_index.get("skipped_sources", [])
                if isinstance(entry, dict) and isinstance(entry.get("source"), dict)
            }
        except Exception:
            prior_skipped_by_path = {}

    shard_entries, skipped_sources, total_events = _resume_prefix_from_index(
        index_path,
        root_sources,
        resume_from_root_index,
    )
    if resume_from_root_index > 1:
        logger.info(
            "Fast resume: trusting %s indexed shard(s) and %s recorded skipped ROOT file(s) "
            "before ROOT index %s without loading shard payloads.",
            len(shard_entries),
            len(skipped_sources),
            resume_from_root_index,
        )
    for shard_idx, source in enumerate(
        root_sources[resume_from_root_index - 1 :],
        start=resume_from_root_index,
    ):
        shard_path = shards_dir / f"shard_{shard_idx:06d}.pt"
        num_events = None
        if skip_existing and shard_path.exists() and not force:
            try:
                num_events = _validate_shard_payload(_load_torch(shard_path), source)
            except Exception:
                num_events = None
            if num_events is not None:
                logger.info("Reusing valid processed shard: %s", shard_path.name)

        if num_events is None:
            prior_skip = prior_skipped_by_path.get(source["path"])
            if prior_skip is not None and prior_skip.get("source") == source:
                skipped_sources.append(prior_skip)
                logger.warning(
                    "Reusing previously recorded skipped ROOT file: %s (%s: %s)",
                    source["name"],
                    prior_skip.get("error_type", "unknown error"),
                    prior_skip.get("error", "no message"),
                )
                _write_json(index_path, _index_payload(shard_entries, skipped_sources, total_events))
                continue

            logger.info("Tensorizing ROOT file into shard: %s -> %s", source["name"], shard_path.name)
            events = []
            try:
                for local_entry, raw_event in iter_ecal_rechits_with_truth_and_triggerpad_context(
                    source["path"],
                    max_events=max_events_per_root_file,
                    step_size=read_step_size,
                ):
                    event = ecal_tpad_event_to_tensors(
                        raw_event,
                        event_idx=total_events + len(events),
                        valid_labels=tuple(valid_labels),
                        target_mode="physical-origin",
                        filter_noise=filter_noise,
                        supervise_noise=supervise_noise,
                    )
                    attach_root_source_metadata(
                        event,
                        {"file": source["name"], "entry": local_entry},
                        global_event_idx=total_events + len(events),
                        electron_count=source["electron_count"],
                        source_label=source["source_label"],
                    )
                    events.append(event)
            except Exception as exc:
                if not skip_failed_root_files:
                    raise
                skipped_source = {
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                skipped_sources.append(skipped_source)
                logger.exception("Skipping failed ROOT file and continuing: %s", source["path"])
                _write_json(index_path, _index_payload(shard_entries, skipped_sources, total_events))
                continue
            payload = {
                "schema_version": SHARD_PAYLOAD_SCHEMA_VERSION,
                "source": source,
                "events": events,
            }
            torch.save(payload, shard_path)
            num_events = len(events)
            logger.info("Wrote %s event(s) to %s", num_events, shard_path)

        entry = {
            "path": str(shard_path.relative_to(cache_dir)).replace("\\", "/"),
            "source": source,
            "num_events": int(num_events),
            "event_start": total_events,
            "event_stop": total_events + int(num_events),
        }
        shard_entries.append(entry)
        total_events += int(num_events)
        _write_json(index_path, _index_payload(shard_entries, skipped_sources, total_events))

    if not shard_entries:
        raise ValueError("No ROOT files were selected for sharded preprocessing.")
    logger.info(
        "Sharded tensor cache ready: %s event(s) in %s shard(s); skipped ROOT files: %s",
        total_events,
        len(shard_entries),
        len(skipped_sources),
    )
    return cache_dir


def validate_sharded_tensor_cache(cache_dir, load_shards=True, allow_incomplete=False):
    """Validate manifest/index presence and optionally load every listed shard."""
    cache_dir = Path(cache_dir)
    if not has_sharded_tensor_cache(cache_dir):
        raise ValueError(f"Sharded cache requires manifest.json and index.json: {cache_dir}")
    manifest = _load_json(cache_dir / "manifest.json")
    index = _load_json(cache_dir / "index.json")
    entries = index.get("shards", [])
    if not entries:
        raise ValueError(f"Sharded cache index contains no shards: {cache_dir}")
    expected_sources = manifest.get("cache_spec", {}).get("root_sources", [])
    skipped_sources = index.get("skipped_sources", [])
    indexed_sources = [entry.get("source") for entry in entries]
    skipped_input_sources = [entry.get("source") for entry in skipped_sources]
    accounted_sources = []
    indexed_idx = 0
    skipped_idx = 0
    for expected_source in expected_sources:
        if indexed_idx < len(indexed_sources) and indexed_sources[indexed_idx] == expected_source:
            accounted_sources.append(expected_source)
            indexed_idx += 1
        elif skipped_idx < len(skipped_input_sources) and skipped_input_sources[skipped_idx] == expected_source:
            accounted_sources.append(expected_source)
            skipped_idx += 1
        else:
            break
    all_index_entries_accounted = indexed_idx == len(indexed_sources) and skipped_idx == len(skipped_input_sources)
    if allow_incomplete:
        sources_match = all_index_entries_accounted
    else:
        sources_match = all_index_entries_accounted and accounted_sources == expected_sources
    if not sources_match:
        raise ValueError(f"Sharded cache index is incomplete or does not match its manifest: {cache_dir}")

    total_events = 0
    for entry in entries:
        shard_path = cache_dir / entry["path"]
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing processed shard listed in index: {shard_path}")
        if entry["event_start"] != total_events:
            raise ValueError(f"Non-contiguous shard event offsets in {cache_dir / 'index.json'}")
        if load_shards:
            payload = _load_torch(shard_path)
            num_events = _validate_shard_payload(payload, entry["source"])
            if num_events is None or num_events != entry["num_events"]:
                raise ValueError(f"Invalid processed shard payload: {shard_path}")
        total_events = entry["event_stop"]
    if total_events != index.get("num_events"):
        raise ValueError(f"Sharded cache event count does not match index: {cache_dir}")
    return manifest, index


class ShardedECalTpadDataset(Dataset):
    """Lazy event dataset backed by one tensor shard per source ROOT file."""

    def __init__(
        self,
        cache_dir,
        max_events=None,
        shard_cache_size=1,
        event_transform=None,
        allow_incomplete=False,
    ):
        if shard_cache_size <= 0:
            raise ValueError("shard_cache_size must be positive.")
        self.cache_dir = Path(cache_dir)
        self.metadata, self.index = validate_sharded_tensor_cache(
            self.cache_dir,
            load_shards=False,
            allow_incomplete=allow_incomplete,
        )
        self.shards = self.index["shards"]
        self.event_stops = [entry["event_stop"] for entry in self.shards]
        available_events = int(self.index["num_events"])
        self.num_events = available_events if max_events is None else min(int(max_events), available_events)
        self.shard_cache_size = int(shard_cache_size)
        self.event_transform = event_transform
        self._loaded_shards = OrderedDict()

    def __len__(self):
        return self.num_events

    @property
    def root_files(self):
        return [Path(entry["source"]["path"]) for entry in self.shards]

    @property
    def source_files(self):
        return [entry["source"]["name"] for entry in self.shards]

    def set_event_transform(self, event_transform):
        self.event_transform = event_transform

    def _shard_idx_for_event(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect_right(self.event_stops, index)

    def _load_shard(self, shard_idx):
        if shard_idx in self._loaded_shards:
            payload = self._loaded_shards.pop(shard_idx)
            self._loaded_shards[shard_idx] = payload
            return payload
        payload = _load_torch(self.cache_dir / self.shards[shard_idx]["path"])
        self._loaded_shards[shard_idx] = payload
        while len(self._loaded_shards) > self.shard_cache_size:
            self._loaded_shards.popitem(last=False)
        return payload

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        shard_idx = self._shard_idx_for_event(index)
        entry = self.shards[shard_idx]
        local_index = index - entry["event_start"]
        event = dict(self._load_shard(shard_idx)["events"][local_index])
        if self.event_transform is not None:
            event = self.event_transform(event)
        return event

    def order_indices_for_access(self, indices, seed=None):
        """Group accesses by shard to avoid repeatedly opening large shard files."""
        grouped = {}
        for index in indices:
            grouped.setdefault(self._shard_idx_for_event(int(index)), []).append(int(index))
        shard_indices = list(grouped)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
            order = torch.randperm(len(shard_indices), generator=generator).tolist()
            shard_indices = [shard_indices[idx] for idx in order]
        else:
            shard_indices.sort()
        ordered = []
        for shard_idx in shard_indices:
            group = grouped[shard_idx]
            if seed is not None and len(group) > 1:
                order = torch.randperm(len(group), generator=generator).tolist()
                group = [group[idx] for idx in order]
            ordered.extend(group)
        return ordered


class MultiShardedECalTpadDataset(Dataset):
    """Lazy view over multiple independent sharded caches."""

    def __init__(self, sources, max_events=None, event_transform=None):
        if not sources:
            raise ValueError("MultiShardedECalTpadDataset requires at least one source.")
        self.sources = list(sources)
        self.event_transform = event_transform
        self.offsets = []
        total = 0
        for source in self.sources:
            self.offsets.append(total)
            total += len(source["dataset"])
        self.num_events = total if max_events is None else min(int(max_events), total)

    def __len__(self):
        return self.num_events

    @property
    def root_files(self):
        files = []
        for source in self.sources:
            files.extend(source["dataset"].root_files)
        return files

    @property
    def source_files(self):
        files = []
        for source in self.sources:
            files.extend(source["dataset"].source_files)
        return files

    @property
    def source_summaries(self):
        return [
            {
                "electron_count": source["electron_count"],
                "source_label": source["source_label"],
                "cache_dir": str(source["cache_dir"]),
                "num_events": len(source["dataset"]),
            }
            for source in self.sources
        ]

    def set_event_transform(self, event_transform):
        self.event_transform = event_transform

    def _source_idx_for_event(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect_right(self.offsets, index) - 1

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        source_idx = self._source_idx_for_event(index)
        source = self.sources[source_idx]
        local_index = index - self.offsets[source_idx]
        event = dict(source["dataset"][local_index])
        if source["electron_count"] is not None:
            event["electron_count"] = torch.tensor(int(source["electron_count"]), dtype=torch.long)
        event["source_label"] = source["source_label"]
        if self.event_transform is not None:
            event = self.event_transform(event)
        return event

    def order_indices_for_access(self, indices, seed=None):
        grouped = {}
        for index in indices:
            source_idx = self._source_idx_for_event(int(index))
            grouped.setdefault(source_idx, []).append(int(index))
        source_indices = list(grouped)
        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(int(seed))
            order = torch.randperm(len(source_indices), generator=generator).tolist()
            source_indices = [source_indices[idx] for idx in order]
        else:
            source_indices.sort()

        ordered = []
        for source_idx in source_indices:
            group = grouped[source_idx]
            local_group = [index - self.offsets[source_idx] for index in group]
            local_order = self.sources[source_idx]["dataset"].order_indices_for_access(
                local_group,
                seed=None if seed is None else int(seed) + source_idx + 1,
            )
            ordered.extend(self.offsets[source_idx] + local_index for local_index in local_order)
        return ordered
