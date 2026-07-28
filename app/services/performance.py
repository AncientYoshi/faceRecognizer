"""Bounded in-process latency and resource metrics."""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from threading import Lock
from time import perf_counter
from typing import ContextManager, Iterator, Protocol

import psutil


@dataclass(frozen=True)
class MetricSummary:
    count: int
    minimum_ms: float
    maximum_ms: float
    average_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


class PerformanceRecorder(Protocol):
    def record(self, metric: str, duration_ms: float) -> None:
        """Record a duration sample."""

    def track(self, metric: str) -> ContextManager[None]:
        """Time a block and record it even when the block raises."""


class NoOpPerformanceRecorder:
    def record(self, metric: str, duration_ms: float) -> None:
        pass

    @contextmanager
    def track(self, metric: str) -> Iterator[None]:
        yield


class PerformanceTracker:
    """Keep recent timings without allowing metrics memory to grow forever."""

    def __init__(self, max_samples_per_metric: int = 1000) -> None:
        if max_samples_per_metric < 1:
            raise ValueError("max_samples_per_metric must be positive.")
        self._max_samples = max_samples_per_metric
        self._samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max_samples_per_metric)
        )
        self._lock = Lock()
        self._started_at = perf_counter()
        self._process = psutil.Process()
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

    @property
    def max_samples_per_metric(self) -> int:
        return self._max_samples

    def record(self, metric: str, duration_ms: float) -> None:
        if duration_ms < 0:
            raise ValueError("A duration cannot be negative.")
        with self._lock:
            self._samples[metric].append(float(duration_ms))

    @contextmanager
    def track(self, metric: str) -> Iterator[None]:
        started_at = perf_counter()
        try:
            yield
        finally:
            self.record(metric, (perf_counter() - started_at) * 1000)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            copied_samples = {
                metric: list(samples)
                for metric, samples in self._samples.items()
            }

        memory = self._process.memory_info()
        metrics = {
            metric: asdict(summarize_durations(values))
            for metric, values in sorted(copied_samples.items())
            if values
        }
        return {
            "capturedAt": datetime.now(UTC).isoformat(),
            "uptimeSeconds": perf_counter() - self._started_at,
            "sampleLimitPerMetric": self._max_samples,
            "metrics": metrics,
            "process": {
                "residentMemoryBytes": memory.rss,
                "virtualMemoryBytes": memory.vms,
                "memoryPercent": self._process.memory_percent(),
                "cpuPercent": self._process.cpu_percent(interval=None),
                "systemCpuPercent": psutil.cpu_percent(interval=None),
                "threadCount": self._process.num_threads(),
            },
        }


def summarize_durations(values: list[float]) -> MetricSummary:
    if not values:
        raise ValueError("At least one duration is required.")
    ordered = sorted(values)
    return MetricSummary(
        count=len(ordered),
        minimum_ms=ordered[0],
        maximum_ms=ordered[-1],
        average_ms=sum(ordered) / len(ordered),
        p50_ms=_percentile(ordered, 0.50),
        p95_ms=_percentile(ordered, 0.95),
        p99_ms=_percentile(ordered, 0.99),
    )


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction
    )
