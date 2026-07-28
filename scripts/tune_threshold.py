"""Evaluate verification thresholds from labeled image pairs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import Settings
from app.repositories.embedding_repository import StoredEmbedding
from app.services.face_service import FaceEmbedding
from app.services.insightface_service import InsightFaceService
from app.services.threshold_tuning import (
    LabeledSimilarity,
    ThresholdEvaluation,
    evaluate_thresholds,
)


class FaceEmbedder(Protocol):
    def generate_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        """Generate a normalized face embedding."""


class CalibrationRepository:
    """No-op repository because calibration only generates embeddings."""

    def initialize(self) -> None:
        pass

    def upsert(
        self,
        student_id: str,
        values: tuple[float, ...],
    ) -> str:
        raise RuntimeError("Calibration does not store student embeddings.")

    def find_by_student_id(self, student_id: str) -> StoredEmbedding | None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate similarities for labeled image pairs and evaluate "
            "candidate verification thresholds."
        )
    )
    parser.add_argument(
        "--pairs",
        required=True,
        type=Path,
        help="CSV containing left_image,right_image,is_match columns.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.40,0.45,0.50,0.55,0.60",
        help="Comma-separated cosine-similarity thresholds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the complete JSON report.",
    )
    return parser.parse_args()


def read_pairs(path: Path) -> list[tuple[Path, Path, bool]]:
    pairs: list[tuple[Path, Path, bool]] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_columns = {"left_image", "right_image", "is_match"}
        if not expected_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "Pair CSV must contain left_image, right_image, and is_match."
            )

        for line_number, row in enumerate(reader, start=2):
            left = _resolve_image_path(path, row["left_image"])
            right = _resolve_image_path(path, row["right_image"])
            is_match = _parse_boolean(row["is_match"], line_number)
            pairs.append((left, right, is_match))

    if not pairs:
        raise ValueError("Pair CSV contains no data rows.")
    return pairs


def build_samples(
    pairs: list[tuple[Path, Path, bool]],
    service: FaceEmbedder,
) -> list[LabeledSimilarity]:
    embedding_cache: dict[Path, np.ndarray] = {}
    samples: list[LabeledSimilarity] = []

    for left_path, right_path, is_match in pairs:
        left = _embedding_for(left_path, service, embedding_cache)
        right = _embedding_for(right_path, service, embedding_cache)
        similarity = float(np.clip(np.dot(left, right), -1.0, 1.0))
        samples.append(
            LabeledSimilarity(similarity=similarity, is_match=is_match)
        )

    return samples


def print_report(evaluation: ThresholdEvaluation) -> None:
    print(
        "threshold  FAR      FRR      accuracy HTER     TP  FP  TN  FN"
    )
    for result in evaluation.metrics:
        marker = "*" if result == evaluation.recommended else " "
        print(
            f"{result.threshold:>8.2f}{marker} "
            f"{result.false_accept_rate:>7.3f}  "
            f"{result.false_reject_rate:>7.3f}  "
            f"{result.accuracy:>7.3f}  "
            f"{result.half_total_error_rate:>7.3f}  "
            f"{result.true_accepts:>2}  "
            f"{result.false_accepts:>2}  "
            f"{result.true_rejects:>2}  "
            f"{result.false_rejects:>2}"
        )
    print(
        f"\nRecommended threshold: {evaluation.recommended.threshold:.2f} "
        "(marked with *)"
    )


def main() -> None:
    args = parse_args()
    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]
    pairs_path = args.pairs.expanduser().resolve()
    pairs = read_pairs(pairs_path)

    service = InsightFaceService(
        Settings(),
        embedding_repository=CalibrationRepository(),
    )
    service.load()
    samples = build_samples(pairs, service)
    evaluation = evaluate_thresholds(samples, thresholds)
    print_report(evaluation)

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "pairCount": len(samples),
                    "recommendedThreshold": evaluation.recommended.threshold,
                    "results": [
                        asdict(result) for result in evaluation.metrics
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _resolve_image_path(csv_path: Path, value: str) -> Path:
    image_path = Path(value.strip()).expanduser()
    if not image_path.is_absolute():
        image_path = csv_path.parent / image_path
    resolved = image_path.resolve()
    if not resolved.is_file():
        raise ValueError(f"Image does not exist: {resolved}")
    return resolved


def _parse_boolean(value: str, line_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(
        f"Invalid is_match value on CSV line {line_number}: {value!r}"
    )


def _embedding_for(
    path: Path,
    service: FaceEmbedder,
    cache: dict[Path, np.ndarray],
) -> np.ndarray:
    if path not in cache:
        result = service.generate_embedding(path.read_bytes())
        cache[path] = np.asarray(result.values, dtype=np.float32)
    return cache[path]


if __name__ == "__main__":
    main()
