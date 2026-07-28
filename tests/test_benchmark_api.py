"""Client benchmark report tests."""

import pytest

from scripts.benchmark_api import RequestResult, summarize_results


def test_summarizes_successful_requests_and_status_counts() -> None:
    results = [
        RequestResult("detect", 200, 10.0, 8.0),
        RequestResult("detect", 200, 20.0, 15.0),
        RequestResult("detect", 422, 5.0, 4.0),
    ]

    summary = summarize_results(results)["detect"]

    assert summary["requestCount"] == 3
    assert summary["statusCounts"] == {"200": 2, "422": 1}
    assert summary["clientLatency"]["average_ms"] == pytest.approx(15.0)
    assert summary["serverLatency"]["average_ms"] == pytest.approx(11.5)
