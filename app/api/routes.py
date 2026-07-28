"""Public HTTP endpoint definitions."""

from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.models.schemas import (
    BoundingBox,
    DetectFaceResponse,
    EmbeddingResponse,
    ErrorResponse,
    HealthResponse,
    PerformanceResponse,
    RegisterFaceResponse,
    VerifyFaceResponse,
)
from app.services.face_service import FaceService
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
