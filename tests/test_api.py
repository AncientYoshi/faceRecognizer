"""Contract tests for the public API."""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.services.face_service import (
    DetectedFace,
    FaceEmbedding,
    FaceIdentification,
    FaceVerification,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)


CANDIDATE_1 = "2f52f06f-59ed-4519-bb86-69cb59fb3197"
CANDIDATE_2 = "12807f44-e4e2-464a-b525-9812b3dc0f3c"


class FakeFaceService:
    def register_face(self, student_id: str, image_bytes: bytes) -> str:
        assert student_id == "STU-001"
        assert image_bytes == b"test-image"
        return "emb-123"

    def verify_face(
        self, student_id: str, image_bytes: bytes
    ) -> FaceVerification:
        assert student_id == "STU-001"
        assert image_bytes == b"test-image"
        return FaceVerification(matched=True, similarity=0.84)

    def identify_face(
        self,
        candidate_student_ids: tuple[str, ...],
        image_bytes: bytes,
    ) -> FaceIdentification:
        assert candidate_student_ids == (CANDIDATE_1, CANDIDATE_2)
        assert image_bytes == b"test-image"
        return FaceIdentification(
            matched=True,
            student_id=CANDIDATE_2,
            similarity=0.93,
            liveness_passed=True,
            reason="MATCHED",
        )

    def detect_face(self, image_bytes: bytes) -> DetectedFace:
        assert image_bytes == b"test-image"
        return DetectedFace(
            bounding_box=(10, 20, 110, 140),
            confidence=0.98,
        )

    def generate_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        assert image_bytes == b"test-image"
        return FaceEmbedding(values=tuple(0.0 for _ in range(512)))


def make_client(
    face_service: FakeFaceService | None = None,
    settings: Settings | None = None,
) -> TestClient:
    app = create_app(
        face_service=face_service or FakeFaceService(),
        settings=settings or Settings(max_upload_size_mb=1),
    )
    return TestClient(app)


def image_file(
    content: bytes = b"test-image",
    content_type: str = "image/jpeg",
) -> dict[str, tuple[str, bytes, str]]:
    return {"image": ("face.jpg", content, content_type)}


def test_health_check() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}
    assert float(response.headers["X-Process-Time-Ms"]) >= 0


def test_performance_metrics_contract() -> None:
    client = make_client()
    client.get("/health")

    response = client.get("/metrics/performance")

    assert response.status_code == 200
    body = response.json()
    assert body["sampleLimitPerMetric"] == 1000
    assert body["metrics"]["http_get_health_ms"]["count"] >= 1
    assert body["process"]["residentMemoryBytes"] > 0


def test_register_face_contract() -> None:
    response = make_client().post(
        "/faces/register",
        data={"studentId": "STU-001"},
        files=image_file(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "embeddingId": "emb-123",
    }


def test_verify_face_contract() -> None:
    response = make_client().post(
        "/faces/verify",
        data={"studentId": "STU-001"},
        files=image_file(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "matched": True,
        "similarity": 0.84,
    }


def test_identify_face_contract() -> None:
    response = make_client().post(
        "/faces/identify",
        data={
            "candidateStudentIds": json.dumps(
                [CANDIDATE_1, CANDIDATE_2]
            )
        },
        files=image_file(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "matched": True,
        "studentId": CANDIDATE_2,
        "similarity": 0.93,
        "livenessPassed": True,
        "reason": "MATCHED",
    }


def test_identify_face_no_match_contract() -> None:
    class NoMatchFaceService(FakeFaceService):
        def identify_face(
            self,
            candidate_student_ids: tuple[str, ...],
            image_bytes: bytes,
        ) -> FaceIdentification:
            return FaceIdentification(
                matched=False,
                student_id=None,
                similarity=0.41,
                liveness_passed=True,
                reason="NOT_MATCHED",
            )

    response = make_client(NoMatchFaceService()).post(
        "/faces/identify",
        data={"candidateStudentIds": json.dumps([CANDIDATE_1])},
        files=image_file(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "matched": False,
        "studentId": None,
        "similarity": 0.41,
        "livenessPassed": True,
        "reason": "NOT_MATCHED",
    }


@pytest.mark.parametrize(
    "candidate_value",
    [
        "not-json",
        "{}",
        "[]",
        '["not-a-uuid"]',
        "[123]",
    ],
)
def test_identify_rejects_invalid_candidate_json(
    candidate_value: str,
) -> None:
    response = make_client().post(
        "/faces/identify",
        data={"candidateStudentIds": candidate_value},
        files=image_file(),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == (
        "invalid_candidate_student_ids"
    )


def test_identify_rejects_too_many_candidates() -> None:
    response = make_client(
        settings=Settings(max_upload_size_mb=1, max_identify_candidates=1)
    ).post(
        "/faces/identify",
        data={
            "candidateStudentIds": json.dumps(
                [CANDIDATE_1, CANDIDATE_2]
            )
        },
        files=image_file(),
    )

    assert response.status_code == 400
    assert "At most 1" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("service_error", "reason"),
    [
        (NoFaceDetectedError("No face."), "NO_FACE_DETECTED"),
        (MultipleFacesDetectedError("Multiple faces."), "MULTIPLE_FACES"),
    ],
)
def test_identify_returns_a_reason_for_face_count_failures(
    service_error: Exception,
    reason: str,
) -> None:
    class FaceCountFailureService(FakeFaceService):
        def identify_face(
            self,
            candidate_student_ids: tuple[str, ...],
            image_bytes: bytes,
        ) -> FaceIdentification:
            raise service_error

    response = make_client(FaceCountFailureService()).post(
        "/faces/identify",
        data={"candidateStudentIds": json.dumps([CANDIDATE_1])},
        files=image_file(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "matched": False,
        "studentId": None,
        "similarity": 0.0,
        "livenessPassed": True,
        "reason": reason,
    }


def test_detect_face_contract() -> None:
    response = make_client().post(
        "/faces/detect",
        files=image_file(),
    )

    assert response.status_code == 200
    assert response.json() == {
        "faceFound": True,
        "boundingBox": {
            "x1": 10,
            "y1": 20,
            "x2": 110,
            "y2": 140,
        },
        "confidence": 0.98,
    }


def test_embedding_contract() -> None:
    response = make_client().post(
        "/faces/embedding",
        files=image_file(),
    )

    assert response.status_code == 200
    assert response.json()["dimension"] == 512
    assert len(response.json()["embedding"]) == 512


def test_rejects_unsupported_image_type() -> None:
    response = make_client().post(
        "/faces/detect",
        files=image_file(content_type="text/plain"),
    )

    assert response.status_code == 415
    assert response.json() == {
        "error": {
            "code": "unsupported_image_type",
            "message": "The image must be JPEG, PNG, or WebP.",
        }
    }


def test_rejects_empty_image() -> None:
    response = make_client().post(
        "/faces/detect",
        files=image_file(content=b""),
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_image"


def test_missing_student_id_has_consistent_error_shape() -> None:
    response = make_client().post(
        "/faces/register",
        files=image_file(),
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_default_service_reports_pipeline_not_ready() -> None:
    app = create_app(settings=Settings(max_upload_size_mb=1))
    response = TestClient(app).post(
        "/faces/detect",
        files=image_file(),
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "pipeline_not_ready"


def test_application_lifespan_loads_a_loadable_service() -> None:
    class LoadableFakeFaceService(FakeFaceService):
        loaded = False

        def load(self) -> None:
            self.loaded = True

    service = LoadableFakeFaceService()
    app = create_app(
        face_service=service,
        settings=Settings(max_upload_size_mb=1),
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert service.loaded is True


def test_unexpected_errors_use_the_consistent_error_envelope() -> None:
    class ExplodingFaceService(FakeFaceService):
        def detect_face(self, image_bytes: bytes) -> DetectedFace:
            raise RuntimeError("sensitive implementation detail")

    app = create_app(
        face_service=ExplodingFaceService(),
        settings=Settings(max_upload_size_mb=1),
    )
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post("/faces/detect", files=image_file())

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "An unexpected internal error occurred.",
        }
    }
