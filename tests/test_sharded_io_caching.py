import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from mldmx.datasets.cached_views import CachedEventViewDataset
from mldmx.datasets.ecal_tpad_loading import load_processed_or_grouped_root_tensor_events
from mldmx.datasets.ecal_tpad_shards import (
    SHARD_CACHE_SCHEMA_VERSION,
    SHARD_PAYLOAD_SCHEMA_VERSION,
    ShardedECalTpadDataset,
)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source(idx):
    return {
        "path": f"/fake/events_{idx}.root",
        "name": f"events_{idx}.root",
        "source_dir": "/fake",
        "source_label": "2e",
        "electron_count": 2,
        "size": 1000 + idx,
        "mtime_ns": 123456 + idx,
    }


def _write_sharded_cache(cache_dir: Path, shard_event_counts):
    shards_dir = cache_dir / "shards"
    shards_dir.mkdir(parents=True)
    sources = [_source(idx + 1) for idx in range(len(shard_event_counts))]
    shard_entries = []
    total_events = 0
    for shard_idx, (source, num_events) in enumerate(zip(sources, shard_event_counts), start=1):
        shard_path = shards_dir / f"shard_{shard_idx:06d}.pt"
        events = [
            {
                "event_idx": torch.tensor(total_events + local_idx, dtype=torch.long),
                "marker": f"shard-{shard_idx}",
            }
            for local_idx in range(num_events)
        ]
        torch.save(
            {
                "schema_version": SHARD_PAYLOAD_SCHEMA_VERSION,
                "source": source,
                "events": events,
            },
            shard_path,
        )
        shard_entries.append(
            {
                "path": str(shard_path.relative_to(cache_dir)),
                "source": source,
                "num_events": num_events,
                "event_start": total_events,
                "event_stop": total_events + num_events,
            }
        )
        total_events += num_events

    _write_json(
        cache_dir / "manifest.json",
        {
            "cache_schema_version": SHARD_CACHE_SCHEMA_VERSION,
            "format": "ecal_tpad_root_file_shards",
            "cache_spec": {
                "root_sources": sources,
            },
        },
    )
    _write_json(
        cache_dir / "index.json",
        {
            "cache_schema_version": SHARD_CACHE_SCHEMA_VERSION,
            "num_events": total_events,
            "shards": shard_entries,
            "skipped_sources": [],
        },
    )


class ShardedIoCachingTest(unittest.TestCase):
    def test_sharded_dataset_reports_lru_cache_hits_and_evictions(self):
        with TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            _write_sharded_cache(cache_dir, [2, 1])
            dataset = ShardedECalTpadDataset(cache_dir, shard_cache_size=1)

            self.assertEqual(dataset[0]["marker"], "shard-1")
            self.assertEqual(dataset[1]["marker"], "shard-1")
            self.assertEqual(dataset[2]["marker"], "shard-2")
            self.assertEqual(dataset[0]["marker"], "shard-1")

            info = dataset.cache_info()
            self.assertEqual(info["cache_hits"], 1)
            self.assertEqual(info["cache_misses"], 3)
            self.assertEqual(info["cache_evictions"], 2)
            self.assertEqual(info["loaded_shards"], [0])
            self.assertEqual(info["shard_event_counts"], [2, 1])

    def test_order_indices_for_access_groups_events_by_shard(self):
        with TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            _write_sharded_cache(cache_dir, [2, 2, 1])
            dataset = ShardedECalTpadDataset(cache_dir, shard_cache_size=1)

            ordered = dataset.order_indices_for_access([4, 0, 3, 1, 2])

            self.assertEqual(ordered, [0, 1, 3, 2, 4])

    def test_processed_loader_prefers_sharded_cache_over_per_event_files(self):
        with TemporaryDirectory() as temporary_dir:
            cache_dir = Path(temporary_dir)
            _write_sharded_cache(cache_dir, [1])
            torch.save({"marker": "event-file"}, cache_dir / "event_000000.pt")

            events, event_sources, data_dir, root_files = load_processed_or_grouped_root_tensor_events(
                processed_dir=cache_dir,
                root_specs=[],
                root_data_dir=cache_dir / "unused-root-dir",
                events_per_source=1,
                max_processed_events=None,
                valid_labels=(1, 2, 3),
            )

            self.assertIsInstance(events, ShardedECalTpadDataset)
            self.assertIs(event_sources, events)
            self.assertEqual(data_dir, cache_dir)
            self.assertEqual([path.name for path in root_files], ["events_1.root"])
            self.assertEqual(events[0]["marker"], "shard-1")

    def test_cached_event_view_dataset_uses_bounded_lru(self):
        calls = {"count": 0}
        base_events = [{"value": idx} for idx in range(3)]

        def view_fn(event):
            calls["count"] += 1
            return {"value": event["value"] * 2}

        dataset = CachedEventViewDataset(base_events, view_fn, max_cache_events=2)

        self.assertEqual(dataset[0]["value"], 0)
        self.assertEqual(dataset[0]["value"], 0)
        self.assertEqual(dataset[1]["value"], 2)
        self.assertEqual(dataset[2]["value"], 4)
        self.assertEqual(dataset[0]["value"], 0)

        info = dataset.cache_info()
        self.assertEqual(calls["count"], 4)
        self.assertEqual(info["cache_hits"], 1)
        self.assertEqual(info["cache_misses"], 4)
        self.assertEqual(info["cache_evictions"], 2)
        self.assertEqual(info["cached_event_indices"], [2, 0])


if __name__ == "__main__":
    unittest.main()
