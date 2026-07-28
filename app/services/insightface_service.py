"""InsightFace detection, alignment, and embedding implementation."""

from __future__ import annotations

import math
from collections.abc import Callable
from threading import Lock
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from app.config import Settings
from app.repositories.embedding_repository import (
    EmbeddingRepository,
    EmbeddingRepositoryError,
    SQLiteEmbeddingRepository,
)
from app.services.face_service import (
    EMBEDDING_DIMENSION,
    DetectedFace,
    EmbeddingError,
    EmbeddingStorageError,
    FaceEmbedding,
    FaceVerification,
    InferenceError,
    InvalidImageError,
    InvalidStudentIdError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
    PoorImageQualityError,
    PipelineLoadError,
    PipelineNotReadyError,
    StudentNotRegisteredError,
)
from app.services.performance import (
    NoOpPerformanceRecorder,
    PerformanceRecorder,
)
from app.utils.images import (
    ImageDecodeError,
    calculate_blur_score,
    decode_image,
)


class FaceAnalyzer(Protocol):
    def prepare(
        self,
        ctx_id: int,
        det_thresh: float,
        det_size: tuple[int, int],
    ) -> None:
        """Prepare ONNX models for inference."""

    def get(
        self,
        image: NDArray[np.uint8],
        max_num: int = 0,
    ) -> list[Any]:
        """Analyze every face in an OpenCV BGR image."""


AnalyzerFactory = Callable[[], FaceAnalyzer]
ImageDecoder = Callable[[bytes], NDArray[np.uint8]]


class InsightFaceService:
    """One process-wide InsightFace model pipeline."""

    def __init__(
        self,
        settings: Settings,
        analyzer_factory: AnalyzerFactory | None = None,
        image_decoder: ImageDecoder | None = None,
        embedding_repository: EmbeddingRepository | None = None,
        performance_recorder: PerformanceRecorder | None = None,
    ) -> None:
        self._settings = settings
        self._analyzer_factory = analyzer_factory or self._create_analyzer
        self._image_decoder = image_decoder or (
            lambda image_bytes: decode_image(
                image_bytes,
                max_pixels=settings.max_image_pixels,
            )
        )
        self._embedding_repository = (
            embedding_repository
            or SQLiteEmbeddingRepository(
                settings.face_database_path,
                settings.face_database_timeout_seconds,
            )
        )
        self._performance = (
            performance_recorder or NoOpPerformanceRecorder()
        )
        self._analyzer: FaceAnalyzer | None = None
        self._storage_ready = False
        self._load_lock = Lock()
        self._inference_lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self._analyzer is not None and self._storage_ready

    def _create_analyzer(self) -> FaceAnalyzer:
        # Keep the heavyweight dependency and possible model download out of
        # module import, API contract tests, and documentation generation.
        from insightface.app import FaceAnalysis

        return FaceAnalysis(
            name=self._settings.insightface_model_name,
            root=self._settings.insightface_model_root,
            allowed_modules=["detection", "recognition"],
            providers=self._settings.insightface_provider_list,
        )

    def load(self) -> None:
        """Load and prepare the model exactly once during application startup."""

        if self.is_ready:
            return

        with self._load_lock:
            if self.is_ready:
                return
            if not self._storage_ready:
                try:
                    self._embedding_repository.initialize()
                except EmbeddingRepositoryError as exc:
                    raise PipelineLoadError(
                        "The face embedding database could not be initialized."
                    ) from exc
                self._storage_ready = True

            if self._analyzer is not None:
                return
            try:
                analyzer = self._analyzer_factory()
                analyzer.prepare(
                    ctx_id=self._settings.insightface_context_id,
                    det_thresh=self._settings.insightface_detection_threshold,
                    det_size=(
                        self._settings.insightface_detection_width,
                        self._settings.insightface_detection_height,
                    ),
                )
            except Exception as exc:
                raise PipelineLoadError(
                    "InsightFace could not load its detection and recognition "
                    "models. Check the model files and ONNX Runtime providers."
                ) from exc

            self._analyzer = analyzer

    def register_face(self, student_id: str, image_bytes: bytes) -> str:
        with self._performance.track("registration_total_ms"):
            self._ensure_ready()
            normalized_student_id = self._normalize_student_id(student_id)
            embedding = self.generate_embedding(image_bytes)
            try:
                with self._performance.track("embedding_store_ms"):
                    return self._embedding_repository.upsert(
                        normalized_student_id,
                        embedding.values,
                    )
            except EmbeddingRepositoryError as exc:
                raise EmbeddingStorageError(
                    "The face embedding could not be stored."
                ) from exc

    def verify_face(
        self,
        student_id: str,
        image_bytes: bytes,
    ) -> FaceVerification:
        with self._performance.track("verification_total_ms"):
            self._ensure_ready()
            normalized_student_id = self._normalize_student_id(student_id)
            try:
                with self._performance.track("embedding_load_ms"):
                    stored = self._embedding_repository.find_by_student_id(
                        normalized_student_id
                    )
            except EmbeddingRepositoryError as exc:
                raise EmbeddingStorageError(
                    "The registered face embedding could not be loaded."
                ) from exc

            if stored is None:
                raise StudentNotRegisteredError(
                    f"Student '{normalized_student_id}' has no registered face."
                )

            candidate = self.generate_embedding(image_bytes)
            stored_values = np.asarray(stored.values, dtype=np.float32)
            candidate_values = np.asarray(
                candidate.values,
                dtype=np.float32,
            )

            if (
                stored_values.size != EMBEDDING_DIMENSION
                or not np.all(np.isfinite(stored_values))
            ):
                raise EmbeddingStorageError(
                    "The registered face embedding is invalid."
                )

            stored_norm = float(np.linalg.norm(stored_values))
            if stored_norm <= 0.0:
                raise EmbeddingStorageError(
                    "The registered face embedding has zero magnitude."
                )

            stored_values /= stored_norm
            similarity = float(np.dot(stored_values, candidate_values))
            similarity = float(np.clip(similarity, -1.0, 1.0))
            return FaceVerification(
                matched=similarity >= self._settings.similarity_threshold,
                similarity=similarity,
            )

    def detect_face(self, image_bytes: bytes) -> DetectedFace:
        with self._performance.track("detection_total_ms"):
            face, image = self._single_face(image_bytes)
            return self._detected_face(face, image)

    def generate_embedding(self, image_bytes: bytes) -> FaceEmbedding:
        with self._performance.track("embedding_generation_total_ms"):
            face, _ = self._single_face(image_bytes)

            with self._performance.track("embedding_postprocess_ms"):
                try:
                    raw_embedding = getattr(face, "normed_embedding", None)
                except Exception as exc:
                    raise EmbeddingError(
                        "InsightFace did not produce a usable face embedding."
                    ) from exc

                if raw_embedding is None:
                    raise EmbeddingError(
                        "The selected InsightFace model pack has no "
                        "recognition output."
                    )

                embedding = np.asarray(
                    raw_embedding,
                    dtype=np.float32,
                ).reshape(-1)
                if embedding.size != EMBEDDING_DIMENSION:
                    raise EmbeddingError(
                        "Expected an embedding with "
                        f"{EMBEDDING_DIMENSION} values, "
                        f"received {embedding.size}."
                    )
                if not np.all(np.isfinite(embedding)):
                    raise EmbeddingError(
                        "The face embedding contains non-finite values."
                    )

                norm = float(np.linalg.norm(embedding))
                if norm <= 0.0:
                    raise EmbeddingError(
                        "The face embedding has zero magnitude."
                    )

                normalized = embedding / norm
                return FaceEmbedding(
                    values=tuple(float(value) for value in normalized)
                )

    @staticmethod
    def _normalize_student_id(student_id: str) -> str:
        normalized = student_id.strip()
        if not normalized:
            raise InvalidStudentIdError("studentId must not be blank.")
        if len(normalized) > 128:
            raise InvalidStudentIdError(
                "studentId must contain at most 128 characters."
            )
        return normalized

    def _ensure_ready(self) -> None:
        if not self.is_ready:
            raise PipelineNotReadyError(
                "The InsightFace pipeline or embedding database "
                "has not completed startup."
            )

    def _single_face(
        self,
        image_bytes: bytes,
    ) -> tuple[Any, NDArray[np.uint8]]:
        analyzer = self._analyzer
        if analyzer is None:
            raise PipelineNotReadyError(
                "The InsightFace pipeline has not completed startup."
            )

        with self._performance.track("image_decode_ms"):
            try:
                image = self._image_decoder(image_bytes)
            except ImageDecodeError as exc:
                raise InvalidImageError(str(exc)) from exc
            except Exception as exc:
                raise InvalidImageError(
                    "The uploaded file could not be decoded as an image."
                ) from exc

        with self._performance.track("image_quality_check_ms"):
            self._validate_image_quality(image)

        try:
            with self._performance.track("insightface_analysis_ms"):
                with self._inference_lock:
                    faces = analyzer.get(image, max_num=0)
        except Exception as exc:
            raise InferenceError("InsightFace inference failed.") from exc

        if not faces:
            raise NoFaceDetectedError(
                "No face was detected. Use a clear, front-facing image."
            )
        if len(faces) > 1:
            raise MultipleFacesDetectedError(
                "Multiple faces were detected. Upload an image containing one face."
            )

        self._validate_face_quality(faces[0], image)
        return faces[0], image

    def _validate_image_quality(self, image: NDArray[np.uint8]) -> None:
        height, width = image.shape[:2]
        if (
            width < self._settings.min_image_width
            or height < self._settings.min_image_height
        ):
            raise PoorImageQualityError(
                "The image is too small. Minimum dimensions are "
                f"{self._settings.min_image_width}x"
                f"{self._settings.min_image_height} pixels."
            )

        minimum_blur_score = self._settings.min_blur_score
        if minimum_blur_score > 0:
            blur_score = calculate_blur_score(image)
            if blur_score < minimum_blur_score:
                raise PoorImageQualityError(
                    "The image appears too blurry. Capture a sharper image "
                    "with the camera held steady."
                )

    def _validate_face_quality(
        self,
        face: Any,
        image: NDArray[np.uint8],
    ) -> None:
        detection = self._detected_face(face, image)
        x1, y1, x2, y2 = detection.bounding_box
        minimum_face_size = self._settings.min_face_size_pixels
        if x2 - x1 < minimum_face_size or y2 - y1 < minimum_face_size:
            raise PoorImageQualityError(
                "The detected face is too small. Move closer to the camera."
            )

    def _detected_face(
        self,
        face: Any,
        image: NDArray[np.uint8],
    ) -> DetectedFace:
        try:
            bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
            confidence = float(face.det_score)
        except Exception as exc:
            raise InferenceError(
                "InsightFace returned an invalid detection result."
            ) from exc

        if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
            raise InferenceError(
                "InsightFace returned an invalid face bounding box."
            )
        if not math.isfinite(confidence):
            raise InferenceError(
                "InsightFace returned an invalid detection confidence."
            )

        height, width = image.shape[:2]
        x1 = max(0, min(width - 1, math.floor(float(bbox[0]))))
        y1 = max(0, min(height - 1, math.floor(float(bbox[1]))))
        x2 = max(0, min(width, math.ceil(float(bbox[2]))))
        y2 = max(0, min(height, math.ceil(float(bbox[3]))))
        if x2 <= x1 or y2 <= y1:
            raise InferenceError(
                "InsightFace returned an empty face bounding box."
            )

        return DetectedFace(
            bounding_box=(x1, y1, x2, y2),
            confidence=confidence,
        )
