"""Minimal Prometheus-text metrics registry for the persona API.

Counters and millisecond histograms, no external dependency. Thread-safe via
a single lock; write volume here is per-request, not per-token.
"""

from __future__ import annotations

import threading
from collections import defaultdict

_BUCKETS_MS = (50, 100, 250, 500, 1000, 3000, 6000, 12000, 30000)


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
        self._histograms: dict[str, list[int]] = {}
        self._histogram_sums: dict[str, float] = defaultdict(float)
        self._histogram_counts: dict[str, int] = defaultdict(int)

    def inc(self, name: str, amount: int = 1, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counter amount must be non-negative")
        key = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._counters[key] += amount

    def observe_ms(self, name: str, value_ms: float) -> None:
        with self._lock:
            buckets = self._histograms.setdefault(name, [0] * (len(_BUCKETS_MS) + 1))
            for index, bound in enumerate(_BUCKETS_MS):
                if value_ms <= bound:
                    buckets[index] += 1
                    break
            else:
                buckets[-1] += 1
            self._histogram_sums[name] += value_ms
            self._histogram_counts[name] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                label_text = ",".join(f'{k}="{v}"' for k, v in labels)
                lines.append(f"{name}{{{label_text}}} {value}" if label_text else f"{name} {value}")
            for name, buckets in sorted(self._histograms.items()):
                cumulative = 0
                for index, bound in enumerate(_BUCKETS_MS):
                    cumulative += buckets[index]
                    lines.append(f'{name}_bucket{{le="{bound}"}} {cumulative}')
                cumulative += buckets[-1]
                lines.append(f'{name}_bucket{{le="+Inf"}} {cumulative}')
                lines.append(f"{name}_sum {self._histogram_sums[name]}")
                lines.append(f"{name}_count {self._histogram_counts[name]}")
        return "\n".join(lines) + "\n"
