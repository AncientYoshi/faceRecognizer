"""Run repeatable client-side benchmarks against the FastAPI service."""

from __future__ import annotations

import argparse
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import httpx

from app.services.performance import summarize_durations


ENDPOINT_PATHS = {
    "detect": "/faces/detect",
    "embedding": "/faces/embedding",
    "verify": "/faces/verify",
}


@dataclass(frozen=True)
class RequestResult:
    endpoint: str
    status_code: int
    client_duration_ms: float
    server_duration_ms: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark face-service latency and fetch process metrics."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--endpoints",
        default="detect,embedding",
        help="Comma-separated selection: detect, embedding, verify.",
    )
    parser.add_argument("--student-id")
    parser.add_argument(
        "--register-image",
        type=Path,
        help="Optionally register student-id before the benchmark.",
    )
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    endpoints = [
        value.strip().lower()
        for value in args.endpoints.split(",")
        if value.strip()
    ]
    unknown = sorted(set(endpoints) - ENDPOINT_PATHS.keys())
    if unknown:
        raise ValueError(f"Unknown benchmark endpoints: {', '.join(unknown)}")
    if not endpoints:
        raise ValueError("At least one endpoint must be selected.")
    if "verify" in endpoints and not args.student_id:
        raise ValueError("--student-id is required when benchmarking verify.")
    if args.requests < 1 or args.warmup < 0 or args.concurrency < 1:
        raise ValueError(
            "requests and concurrency must be positive; warmup cannot be negative."
        )

    image_path = args.image.expanduser().resolve()
    image_bytes = image_path.read_bytes()
    filename = image_path.name
    content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    base_url = args.base_url.rstrip("/")

    with httpx.Client(base_url=base_url, timeout=args.timeout) as client:
        if args.register_image:
            if not args.student_id:
                raise ValueError(
                    "--student-id is required with --register-image."
                )
            register_path = args.register_image.expanduser().resolve()
            register_response = client.post(
                "/faces/register",
                data={"studentId": args.student_id},
                files={
                    "image": (
                        register_path.name,
                        register_path.read_bytes(),
                        mimetypes.guess_type(register_path.name)[0]
                        or "image/jpeg",
                    )
                },
            )
            register_response.raise_for_status()

        for _ in range(args.warmup):
            for endpoint in endpoints:
                _request_endpoint(
                    client,
                    endpoint,
                    filename,
                    image_bytes,
                    content_type,
                    args.student_id,
                )

        jobs = [
            endpoint
            for endpoint in endpoints
            for _ in range(args.requests)
        ]
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            results = list(
                executor.map(
                    lambda endpoint: _request_endpoint(
                        client,
                        endpoint,
                        filename,
                        image_bytes,
                        content_type,
                        args.student_id,
                    ),
                    jobs,
                )
            )

        metrics_response = client.get("/metrics/performance")
        metrics_response.raise_for_status()

    return {
        "configuration": {
            "baseUrl": base_url,
            "endpoints": endpoints,
            "requestsPerEndpoint": args.requests,
            "warmupPerEndpoint": args.warmup,
            "concurrency": args.concurrency,
        },
        "clientResults": summarize_results(results),
        "serviceMetrics": metrics_response.json(),
    }


def summarize_results(
    results: list[RequestResult],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for endpoint in sorted({result.endpoint for result in results}):
        endpoint_results = [
            result for result in results if result.endpoint == endpoint
        ]
        successful = [
            result for result in endpoint_results if result.status_code < 400
        ]
        status_counts: dict[str, int] = {}
        for result in endpoint_results:
            key = str(result.status_code)
            status_counts[key] = status_counts.get(key, 0) + 1

        endpoint_summary: dict[str, object] = {
            "requestCount": len(endpoint_results),
            "statusCounts": status_counts,
        }
        if successful:
            endpoint_summary["clientLatency"] = asdict(
                summarize_durations(
                    [result.client_duration_ms for result in successful]
                )
            )
            server_values = [
                result.server_duration_ms
                for result in successful
                if result.server_duration_ms is not None
            ]
            if server_values:
                endpoint_summary["serverLatency"] = asdict(
                    summarize_durations(server_values)
                )
        summary[endpoint] = endpoint_summary

    return summary


def print_summary(report: dict[str, object]) -> None:
    results = report["clientResults"]
    assert isinstance(results, dict)
    print("endpoint    count  avg ms   p50 ms   p95 ms   p99 ms   statuses")
    for endpoint, raw_summary in results.items():
        summary = raw_summary
        assert isinstance(summary, dict)
        latency = summary.get("clientLatency", {})
        assert isinstance(latency, dict)
        statuses = summary["statusCounts"]
        print(
            f"{endpoint:<11}"
            f"{summary['requestCount']:>5}  "
            f"{latency.get('average_ms', 0):>7.2f}  "
            f"{latency.get('p50_ms', 0):>7.2f}  "
            f"{latency.get('p95_ms', 0):>7.2f}  "
            f"{latency.get('p99_ms', 0):>7.2f}   "
            f"{statuses}"
        )


def _request_endpoint(
    client: httpx.Client,
    endpoint: str,
    filename: str,
    image_bytes: bytes,
    content_type: str,
    student_id: str | None,
) -> RequestResult:
    data = {"studentId": student_id} if endpoint == "verify" else None
    started_at = perf_counter()
    response = client.post(
        ENDPOINT_PATHS[endpoint],
        data=data,
        files={"image": (filename, image_bytes, content_type)},
    )
    duration_ms = (perf_counter() - started_at) * 1000
    server_header = response.headers.get("X-Process-Time-Ms")
    return RequestResult(
        endpoint=endpoint,
        status_code=response.status_code,
        client_duration_ms=duration_ms,
        server_duration_ms=(
            float(server_header) if server_header is not None else None
        ),
    )


def main() -> None:
    args = parse_args()
    report = run_benchmark(args)
    print_summary(report)

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
