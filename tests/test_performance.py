"""Bounded performance tracking tests."""

import pytest

from app.services.performance import (
    PerformanceTracker,
    summarize_durations,
)


def test_summarizes_latency_percentiles() -> None:
    summary = summarize_durations([10.0, 20.0, 30.0, 40.0])

    assert summary.count == 4
    assert summary.minimum_ms == 10.0
    assert summary.maximum_ms == 40.0
    assert summary.average_ms == 25.0
    assert summary.p50_ms == 25.0
    assert summary.p95_ms == pytest.approx(38.5)


def test_tracker_keeps_only_the_configured_recent_samples() -> None:
    tracker = PerformanceTracker(max_samples_per_metric=3)
    for value in [10.0, 20.0, 30.0, 40.0]:
        tracker.record("inference_ms", value)

    snapshot = tracker.snapshot()
    summary = snapshot["metrics"]["inference_ms"]  # type: ignore[index]

    assert summary["count"] == 3
    assert summary["minimum_ms"] == 20.0
    assert summary["maximum_ms"] == 40.0
    assert snapshot["process"]["residentMemoryBytes"] > 0  # type: ignore[index]
