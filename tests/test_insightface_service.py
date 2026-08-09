"""Unit tests for the model pipeline without downloading model weights."""

from typing import Any

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.repositories.embedding_repository import StoredEmbedding
from app.services.face_service import (
    EmbeddingError,
    InvalidImageError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    PoorImageQualityError,
    StudentNotRegisteredError,
)
from app.services.insightface_service import InsightFaceService
from app.services.performance import PerformanceTracker


class FakeFace:
    def __init__(
        self,
        *,
        bbox: tuple[float, float, float, float] = (10.2, 20.7, 110.1, 90.2),
        det_score: float = 0.98,
        embedding: np.ndarray[Any, np.dtype[np.float32]] | None = None,
    ) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.det_score = det_score
        self.normed_embedding = (
            embedding
            if embedding is not None
            else np.ones(512, dtype=np.float32)
        )


class FakeAnalyzer:
    def __init__(self, faces: list[FakeFace]) -> None:
        self.faces = faces
        self.prepare_arguments: dict[str, object] | None = None
        self.received_max_num: int | None = None

    def prepare(
        self,
        ctx_id: int,
        det_thresh: float,
        det_size: tuple[int, int],
    ) -> None:
        self.prepare_arguments = {
            "ctx_id": ctx_id,
            "det_thresh": det_thresh,
            "det_size": det_size,
        }

    def get(
        self,
        image: np.ndarray[Any, np.dtype[np.uint8]],
        max_num: int = 0,
    ) -> list[FakeFace]:
        assert image.shape == (100, 200, 3)
        self.received_max_num = max_num
        return self.faces


class FakeEmbeddingRepository:
    def __init__(self) -> None:
        self.initialized = False
        self.records: dict[str, StoredEmbedding] = {}
        self._next_id = 1
        self.last_candidate_ids: tuple[str, ...] | None = None

    def initialize(self) -> None:
        self.initialized = True

    def upsert(
        self,
        student_id: str,
        values: tuple[float, ...],
    ) -> str:
        embedding_id = f"emb-{self._next_id}"
        self._next_id += 1
        self.records[student_id] = StoredEmbedding(
            student_id=student_id,
            embedding_id=embedding_id,
            values=values,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
        return embedding_id

    def find_by_student_id(self, student_id: str) -> StoredEmbedding | None:
        return self.records.get(student_id)

    def find_by_student_ids(
        self,
        student_ids: tuple[str, ...],
    ) -> list[StoredEmbedding]:
        self.last_candidate_ids = student_ids
        return [
            self.records[student_id]
            for student_id in student_ids
            if student_id in self.records
        ]


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "insightface_context_id": 0,
        "insightface_detection_threshold": 0.6,
        "insightface_detection_width": 640,
        "insightface_detection_height": 640,
        "min_image_width": 1,
        "min_image_height": 1,
        "min_face_size_pixels": 1,
        "min_blur_score": 0,
    }
    values.update(overrides)
    return Settings(**values)


def service_with(
    faces: list[FakeFace],
    *,
    use_real_decoder: bool = False,
    repository: FakeEmbeddingRepository | None = None,
    app_settings: Settings | None = None,
    performance_tracker: PerformanceTracker | None = None,
) -> tuple[InsightFaceService, FakeAnalyzer, FakeEmbeddingRepository]:
    analyzer = FakeAnalyzer(faces)
    embedding_repository = repository or FakeEmbeddingRepository()
    decoder = None
    if not use_real_decoder:
        decoder = lambda _: np.zeros((100, 200, 3), dtype=np.uint8)
    service = InsightFaceService(
        app_settings or settings(),
        analyzer_factory=lambda: analyzer,
        image_decoder=decoder,
        embedding_repository=embedding_repository,
        performance_recorder=performance_tracker,
    )
    service.load()
    return service, analyzer, embedding_repository


def test_load_prepares_the_analyzer_once() -> None:
    service, analyzer, repository = service_with([FakeFace()])

    service.load()

    assert service.is_ready is True
    assert analyzer.prepare_arguments == {
        "ctx_id": 0,
        "det_thresh": 0.6,
        "det_size": (640, 640),
    }
    assert repository.initialized is True


def test_detects_one_face_and_clamps_its_bounding_box() -> None:
    service, analyzer, _ = service_with(
        [FakeFace(bbox=(-3.4, 20.7, 205.2, 90.2))]
    )

    result = service.detect_face(b"image")

    assert result.bounding_box == (0, 20, 200, 91)
    assert result.confidence == pytest.approx(0.98)
    assert analyzer.received_max_num == 0


def test_returns_a_normalized_512_dimension_embedding() -> None:
    service, _, _ = service_with(
        [FakeFace(embedding=np.full(512, 2.0, dtype=np.float32))]
    )

    result = service.generate_embedding(b"image")

    assert result.dimension == 512
    assert np.linalg.norm(result.values) == pytest.approx(1.0)
    assert all(np.isfinite(result.values))


def test_records_model_and_embedding_timings() -> None:
    tracker = PerformanceTracker(max_samples_per_metric=10)
    service, _, _ = service_with(
        [FakeFace()],
        performance_tracker=tracker,
    )

    service.generate_embedding(b"image")
    metrics = tracker.snapshot()["metrics"]

    assert metrics["image_decode_ms"]["count"] == 1
    assert metrics["insightface_analysis_ms"]["count"] == 1
    assert metrics["embedding_postprocess_ms"]["count"] == 1
    assert metrics["embedding_generation_total_ms"]["count"] == 1


def test_rejects_no_face() -> None:
    service, _, _ = service_with([])

    with pytest.raises(NoFaceDetectedError):
        service.detect_face(b"image")


def test_rejects_multiple_faces() -> None:
    service, _, _ = service_with([FakeFace(), FakeFace()])

    with pytest.raises(MultipleFacesDetectedError):
        service.generate_embedding(b"image")


def test_rejects_an_unexpected_embedding_dimension() -> None:
    service, _, _ = service_with(
        [FakeFace(embedding=np.ones(128, dtype=np.float32))]
    )

    with pytest.raises(EmbeddingError, match="512 values"):
        service.generate_embedding(b"image")


def test_real_decoder_rejects_corrupted_image_bytes() -> None:
    service, _, _ = service_with([FakeFace()], use_real_decoder=True)

    with pytest.raises(InvalidImageError):
        service.detect_face(b"not-an-image")


def test_real_decoder_accepts_an_encoded_color_image() -> None:
    service, _, _ = service_with([FakeFace()], use_real_decoder=True)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok

    result = service.detect_face(encoded.tobytes())

    assert result.bounding_box == (10, 20, 111, 91)


def test_rejects_an_image_below_the_minimum_dimensions() -> None:
    service, analyzer, _ = service_with(
        [FakeFace()],
        app_settings=settings(min_image_height=101),
    )

    with pytest.raises(PoorImageQualityError, match="too small"):
        service.detect_face(b"image")

    assert analyzer.received_max_num is None


def test_rejects_a_blurry_image_before_inference() -> None:
    service, analyzer, _ = service_with(
        [FakeFace()],
        app_settings=settings(min_blur_score=10.0),
    )

    with pytest.raises(PoorImageQualityError, match="blurry"):
        service.generate_embedding(b"image")

    assert analyzer.received_max_num is None


def test_rejects_a_face_that_is_too_small() -> None:
    service, _, _ = service_with(
        [FakeFace(bbox=(10.0, 10.0, 29.0, 29.0))],
        app_settings=settings(min_face_size_pixels=20),
    )

    with pytest.raises(PoorImageQualityError, match="face is too small"):
        service.detect_face(b"image")


def test_registers_and_verifies_the_same_face() -> None:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    service, analyzer, repository = service_with(
        [FakeFace(embedding=embedding)],
        app_settings=settings(similarity_threshold=0.5),
    )

    embedding_id = service.register_face(" STU-001 ", b"registration")
    analyzer.faces = [FakeFace(embedding=embedding)]
    verification = service.verify_face("STU-001", b"verification")

    assert embedding_id == "emb-1"
    assert repository.records["STU-001"].embedding_id == embedding_id
    assert verification.matched is True
    assert verification.similarity == pytest.approx(1.0)


def test_verification_rejects_a_different_face() -> None:
    registered = np.zeros(512, dtype=np.float32)
    registered[0] = 1.0
    different = np.zeros(512, dtype=np.float32)
    different[1] = 1.0
    service, analyzer, _ = service_with(
        [FakeFace(embedding=registered)],
        app_settings=settings(similarity_threshold=0.5),
    )

    service.register_face("STU-001", b"registration")
    analyzer.faces = [FakeFace(embedding=different)]
    verification = service.verify_face("STU-001", b"verification")

    assert verification.matched is False
    assert verification.similarity == pytest.approx(0.0)


def test_registration_replaces_the_students_previous_embedding() -> None:
    first = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    replacement = np.zeros(512, dtype=np.float32)
    replacement[1] = 1.0
    service, analyzer, repository = service_with(
        [FakeFace(embedding=first)]
    )

    first_id = service.register_face("STU-001", b"first")
    analyzer.faces = [FakeFace(embedding=replacement)]
    replacement_id = service.register_face("STU-001", b"replacement")

    assert replacement_id != first_id
    assert repository.records["STU-001"].embedding_id == replacement_id
    assert repository.records["STU-001"].values[1] == pytest.approx(1.0)


def test_verification_rejects_an_unregistered_student_before_inference() -> None:
    service, analyzer, _ = service_with([FakeFace()])

    with pytest.raises(StudentNotRegisteredError):
        service.verify_face("UNKNOWN", b"image")

    assert analyzer.received_max_num is None


def test_identifies_the_best_registered_candidate() -> None:
    candidate_1 = "2f52f06f-59ed-4519-bb86-69cb59fb3197"
    candidate_2 = "12807f44-e4e2-464a-b525-9812b3dc0f3c"
    first = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    second = np.zeros(512, dtype=np.float32)
    second[1] = 1.0
    repository = FakeEmbeddingRepository()
    repository.upsert(candidate_1, tuple(float(value) for value in first))
    repository.upsert(candidate_2, tuple(float(value) for value in second))
    service, _, _ = service_with(
        [FakeFace(embedding=second)],
        repository=repository,
        app_settings=settings(similarity_threshold=0.5),
    )

    result = service.identify_face(
        (candidate_1, candidate_2),
        b"image",
    )

    assert result.matched is True
    assert result.student_id == candidate_2
    assert result.similarity == pytest.approx(1.0)
    assert result.liveness_passed is True
    assert result.reason == "MATCHED"
    assert repository.last_candidate_ids == (candidate_1, candidate_2)


def test_identification_returns_no_match_for_unregistered_candidates() -> None:
    candidate_id = "2f52f06f-59ed-4519-bb86-69cb59fb3197"
    service, analyzer, repository = service_with([FakeFace()])

    result = service.identify_face((candidate_id,), b"image")

    assert result.matched is False
    assert result.student_id is None
    assert result.similarity == 0.0
    assert result.reason == "NOT_MATCHED"
    assert repository.last_candidate_ids == (candidate_id,)
    assert analyzer.received_max_num == 0


def test_identification_clamps_a_negative_similarity_to_zero() -> None:
    candidate_id = "2f52f06f-59ed-4519-bb86-69cb59fb3197"
    registered = np.zeros(512, dtype=np.float32)
    registered[0] = 1.0
    presented = np.zeros(512, dtype=np.float32)
    presented[0] = -1.0
    repository = FakeEmbeddingRepository()
    repository.upsert(
        candidate_id,
        tuple(float(value) for value in registered),
    )
    service, _, _ = service_with(
        [FakeFace(embedding=presented)],
        repository=repository,
    )

    result = service.identify_face((candidate_id,), b"image")

    assert result.matched is False
    assert result.student_id is None
    assert result.similarity == 0.0
    assert result.reason == "NOT_MATCHED"
