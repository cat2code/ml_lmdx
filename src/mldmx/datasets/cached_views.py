"""Lazy event-view caching helpers for training loops."""

from collections import OrderedDict


class CachedEventViewDataset:
    """
    Dataset wrapper that caches ``view_fn(event)`` results by event index.

    This keeps the underlying event dataset responsible for IO and physics
    transforms, while avoiding repeated view construction and normalization
    clones on revisits. ``max_cache_events=None`` means unbounded caching;
    otherwise the cache is an event-count LRU.
    """

    def __init__(self, events, view_fn, max_cache_events=None):
        if max_cache_events is not None and int(max_cache_events) <= 0:
            raise ValueError("max_cache_events must be positive or None for unbounded caching.")
        self.events = events
        self.view_fn = view_fn
        self.max_cache_events = None if max_cache_events is None else int(max_cache_events)
        self._cache = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index in self._cache:
            self._hits += 1
            view = self._cache.pop(index)
            self._cache[index] = view
            return view

        self._misses += 1
        view = self.view_fn(self.events[index])
        self._cache[index] = view
        if self.max_cache_events is not None:
            while len(self._cache) > self.max_cache_events:
                self._cache.popitem(last=False)
                self._evictions += 1
        return view

    def order_indices_for_access(self, indices, seed=None):
        if hasattr(self.events, "order_indices_for_access"):
            return self.events.order_indices_for_access(indices, seed=seed)
        return list(indices)

    @property
    def source_files(self):
        return self.events.source_files if hasattr(self.events, "source_files") else []

    @property
    def root_files(self):
        return self.events.root_files if hasattr(self.events, "root_files") else []

    @property
    def source_summaries(self):
        return self.events.source_summaries if hasattr(self.events, "source_summaries") else []

    def clear_cache(self):
        self._cache.clear()

    def cache_info(self):
        base_info = self.events.cache_info() if hasattr(self.events, "cache_info") else None
        return {
            "kind": "event_view",
            "num_events": len(self),
            "max_cache_events": self.max_cache_events,
            "cached_events": len(self._cache),
            "cache_hits": int(self._hits),
            "cache_misses": int(self._misses),
            "cache_evictions": int(self._evictions),
            "cached_event_indices": list(self._cache.keys()),
            "base_cache": base_info,
        }
