"""Public HTTP endpoint definitions."""

import json
from typing import Annotated, cast
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.models.schemas import (
    BoundingBox,
    DetectFaceResponse,
    EmbeddingResponse,
    ErrorResponse,
    HealthResponse,
    IdentifyFaceResponse,
    PerformanceResponse,
    RegisterFaceResponse,
    VerifyFaceResponse,
)
from app.services.face_service import (
    FaceService,
    IdentifyReason,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from app.services.performance import PerformanceTracker
from app.utils.uploads import read_image_upload


router = APIRouter()

ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def get_face_service(request: Request) -> FaceService:
    return cast(FaceService, request.app.state.face_service)


def get_app_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_performance_tracker(request: Request) -> PerformanceTracker:
    return cast(PerformanceTracker, request.app.state.performance_tracker)


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Check service health",
)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get(
    "/metrics/performance",
    response_model=PerformanceResponse,
    tags=["system"],
    summary="Inspect recent latency and process resource metrics",
)
async def performance_metrics(
    tracker: Annotated[
        PerformanceTracker,
        Depends(get_performance_tracker),
    ],
) -> PerformanceResponse:
    return PerformanceResponse.model_validate(tracker.snapshot())


@router.post(
    "/faces/register",
    response_model=RegisterFaceResponse,
    responses=ERROR_RESPONSES,
    tags=["faces"],
    summary="Register a student's face",
)
async def register_face(
    student_id: Annotated[
        str,
        Form(alias="studentId", min_length=1, max_length=128),
    ],
    image: Annotated[UploadFile, File()],
    service: Annotated[FaceService, Depends(get_face_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> RegisterFaceResponse:
    image_bytes = await read_image_upload(image, settings.max_upload_size_bytes)
    embedding_id = await run_in_threadpool(
        service.register_face, student_id, image_bytes
    )
    return RegisterFaceResponse(success=True, embedding_id=embedding_id)


@router.post(
    "/faces/verify",
    response_model=VerifyFaceResponse,
    responses=ERROR_RESPONSES,
    tags=["faces"],
    summary="Verify a student's face",
)
async def verify_face(
    student_id: Annotated[
        str,
        Form(alias="studentId", min_length=1, max_length=128),
    ],
    image: Annotated[UploadFile, File()],
    service: Annotated[FaceService, Depends(get_face_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> VerifyFaceResponse:
    image_bytes = await read_image_upload(image, settings.max_upload_size_bytes)
    result = await run_in_threadpool(
        service.verify_face, student_id, image_bytes
    )
    return VerifyFaceResponse(
        matched=result.matched,
        similarity=result.similarity,
    )


@router.post(
    "/faces/identify",
    response_model=IdentifyFaceResponse,
    responses=ERROR_RESPONSES,
    tags=["faces"],
    summary="Identify a face among supplied candidate students",
)
async def identify_face(
    candidate_student_ids_json: Annotated[
        str,
        Form(alias="candidateStudentIds"),
    ],
    image: Annotated[UploadFile, File()],
    service: Annotated[FaceService, Depends(get_face_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> IdentifyFaceResponse:
    candidate_ids = _parse_candidate_student_ids(
        candidate_student_ids_json,
        settings.max_identify_candidates,
    )
    image_bytes = await read_image_upload(image, settings.max_upload_size_bytes)
    try:
        result = await run_in_threadpool(
            service.identify_face,
            candidate_ids,
            image_bytes,
        )
    except NoFaceDetectedError:
        return _unmatched_identification("NO_FACE_DETECTED")
    except MultipleFacesDetectedError:
        return _unmatched_identification("MULTIPLE_FACES")

    return IdentifyFaceResponse(
        matched=result.matched,
        student_id=result.student_id,
        similarity=result.similarity,
        liveness_passed=result.liveness_passed,
        reason=result.reason,
    )


def _parse_candidate_student_ids(
    raw_value: str,
    maximum_candidates: int,
) -> tuple[str, ...]:
    try:
        decoded = json.loads(raw_value)
        if not isinstance(decoded, list):
            raise TypeError
        if not decoded:
            raise ValueError("At least one candidate is required.")
        if len(decoded) > maximum_candidates:
            raise ValueError(
                f"At most {maximum_candidates} candidates are allowed."
            )
        if any(not isinstance(value, str) for value in decoded):
            raise TypeError
        return tuple(
            dict.fromkeys(str(UUID(value)) for value in decoded)
        )
    except json.JSONDecodeError as exc:
        raise _candidate_ids_error(
            "candidateStudentIds must be a JSON UUID array."
        ) from exc
    except TypeError as exc:
        raise _candidate_ids_error(
            "candidateStudentIds must be a JSON UUID array."
        ) from exc
    except ValueError as exc:
        message = str(exc)
        if message.startswith("At least") or message.startswith("At most"):
            raise _candidate_ids_error(message) from exc
        raise _candidate_ids_error(
            "candidateStudentIds must be a JSON UUID array."
        ) from exc


def _candidate_ids_error(message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "code": "invalid_candidate_student_ids",
            "message": message,
        },
    )


def _unmatched_identification(
    reason: IdentifyReason,
) -> IdentifyFaceResponse:
    return IdentifyFaceResponse(
        matched=False,
        student_id=None,
        similarity=0.0,
        liveness_passed=True,
        reason=reason,
    )


@router.post(
    "/faces/detect",
    response_model=DetectFaceResponse,
    responses=ERROR_RESPONSES,
    tags=["faces"],
    summary="Detect one face in an image",
)
async def detect_face(
    image: Annotated[UploadFile, File()],
    service: Annotated[FaceService, Depends(get_face_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> DetectFaceResponse:
    image_bytes = await read_image_upload(image, settings.max_upload_size_bytes)
    result = await run_in_threadpool(service.detect_face, image_bytes)
    x1, y1, x2, y2 = result.bounding_box
    return DetectFaceResponse(
        face_found=True,
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        confidence=result.confidence,
    )


@router.post(
    "/faces/embedding",
    response_model=EmbeddingResponse,
    responses=ERROR_RESPONSES,
    tags=["faces"],
    summary="Generate a normalized face embedding",
)
async def generate_embedding(
    image: Annotated[UploadFile, File()],
    service: Annotated[FaceService, Depends(get_face_service)],
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> EmbeddingResponse:
    image_bytes = await read_image_upload(image, settings.max_upload_size_bytes)
    result = await run_in_threadpool(service.generate_embedding, image_bytes)
    return EmbeddingResponse(
        dimension=result.dimension,
        embedding=list(result.values),
    )
