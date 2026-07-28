"""Tests for the image-pair calibration workflow."""

from pathlib import Path

import pytest

from app.services.face_service import FaceEmbedding
from scripts.tune_threshold import build_samples, read_pairs


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls: list[bytes] = []

    def generate_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        self.calls.append(image_bytes)
        values = [0.0] * 512
        values[0 if image_bytes == b"first" else 1] = 1.0
        return FaceEmbedding(values=tuple(values))


def test_reads_relative_pair_paths_and_caches_unique_embeddings(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    pairs_csv = tmp_path / "pairs.csv"
    pairs_csv.write_text(
        "left_image,right_image,is_match\n"
        "first.jpg,first.jpg,true\n"
        "first.jpg,second.jpg,false\n",
        encoding="utf-8",
    )
    service = FakeEmbeddingService()

    pairs = read_pairs(pairs_csv)
    samples = build_samples(pairs, service)

    assert [sample.is_match for sample in samples] == [True, False]
    assert samples[0].similarity == pytest.approx(1.0)
    assert samples[1].similarity == pytest.approx(0.0)
    assert service.calls == [b"first", b"second"]


def test_rejects_an_invalid_pair_label(tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    first.write_bytes(b"first")
    pairs_csv = tmp_path / "pairs.csv"
    pairs_csv.write_text(
        "left_image,right_image,is_match\n"
        "first.jpg,first.jpg,maybe\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid is_match"):
        read_pairs(pairs_csv)
