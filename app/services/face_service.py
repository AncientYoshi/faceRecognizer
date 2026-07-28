"""Face-service contracts, results, and domain errors."""

from dataclasses import dataclass
from typing import Protocol


EMBEDDING_DIMENSION = 512


@dataclass(frozen=True)
class DetectedFace:
    bounding_box: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class FaceVerification:
    matched: bool
    similarity: float


@dataclass(frozen=True)
class FaceEmbedding:
    values: tuple[float, ...]

    @property
    def dimension(self) -> int:
        return len(self.values)


class FaceService(Protocol):
    """Boundary implemented by the InsightFace pipeline."""

    def register_face(self, student_id: str, image_bytes: bytes) -> str:
        """Register a face and return its embedding identifier."""

    def verify_face(
        self, student_id: str, image_bytes: bytes
    ) -> FaceVerification:
        """Compare an uploaded face with a registered face."""

    def detect_face(self, image_bytes: bytes) -> DetectedFace:
        """Return the single detected face."""

    def generate_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        """Return the normalized embedding for the single detected face."""


class FaceServiceError(Exception):
    """Base exception converted into a stable HTTP error response."""

    status_code = 500
    code = "face_service_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PipelineNotReadyError(FaceServiceError):
    status_code = 503
    code = "pipeline_not_ready"


class PipelineLoadError(FaceServiceError):
    status_code = 503
    code = "pipeline_load_failed"


class InvalidImageError(FaceServiceError):
    status_code = 400
    code = "invalid_image"


class PoorImageQualityError(FaceServiceError):
    status_code = 422
    code = "poor_image_quality"


class NoFaceDetectedError(FaceServiceError):
    status_code = 422
    code = "no_face_detected"


class MultipleFacesDetectedError(FaceServiceError):
    status_code = 422
    code = "multiple_faces_detected"


class InferenceError(FaceServiceError):
    status_code = 500
    code = "inference_failed"


class EmbeddingError(FaceServiceError):
    status_code = 500
    code = "invalid_embedding"


class InvalidStudentIdError(FaceServiceError):
    status_code = 422
    code = "invalid_student_id"


class StudentNotRegisteredError(FaceServiceError):
    status_code = 404
    code = "student_not_registered"


class EmbeddingStorageError(FaceServiceError):
    status_code = 500
    code = "embedding_storage_failed"


class UnavailableFaceService:
    """Fallback useful when an application explicitly disables the pipeline."""

    _message = "The InsightFace pipeline has not been loaded."

    def register_face(self, student_id: str, image_bytes: bytes) -> str:
        raise PipelineNotReadyError(self._message)

    def verify_face(
        self, student_id: str, image_bytes: bytes
    ) -> FaceVerification:
        raise PipelineNotReadyError(self._message)

    def detect_face(self, image_bytes: bytes) -> DetectedFace:
        raise PipelineNotReadyError(self._message)

    def generate_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        raise PipelineNotReadyError(self._message)
