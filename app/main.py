"""FastAPI application factory."""

from contextlib import asynccontextmanager
import logging
from time import perf_counter
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.routes import router
from app.config import Settings, get_settings
from app.models.schemas import ErrorDetail, ErrorResponse
from app.services.face_service import (
    FaceService,
    FaceServiceError,
)
from app.services.insightface_service import InsightFaceService
from app.services.performance import PerformanceTracker


logger = logging.getLogger(__name__)


def _error_response(
    status_code: int, code: str, message: str
) -> JSONResponse:
    body = ErrorResponse(error=ErrorDetail(code=code, message=message))
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(by_alias=True),
    )


async def face_service_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    error = cast(FaceServiceError, exc)
    return _error_response(error.status_code, error.code, error.message)


async def http_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    error = cast(HTTPException, exc)
    if isinstance(error.detail, dict):
        code = str(error.detail.get("code", "http_error"))
        message = str(error.detail.get("message", "The request failed."))
    else:
        code = "http_error"
        message = str(error.detail)
    return _error_response(error.status_code, code, message)


async def validation_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    error = cast(RequestValidationError, exc)
    errors = error.errors()
    message = errors[0]["msg"] if errors else "The request is invalid."
    return _error_response(422, "validation_error", message)


async def unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.error(
        "Unhandled error while processing %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return _error_response(
        500,
        "internal_error",
        "An unexpected internal error occurred.",
    )


def create_app(
    face_service: FaceService | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Build an application, allowing tests to inject a lightweight engine."""

    app_settings = settings or get_settings()
    performance_tracker = PerformanceTracker(
        app_settings.performance_sample_limit
    )
    service = face_service or InsightFaceService(
        app_settings,
        performance_recorder=performance_tracker,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        load = getattr(application.state.face_service, "load", None)
        if callable(load):
            await run_in_threadpool(load)
        yield

    application = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=(
            "Independent face detection, registration, and verification service."
        ),
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.state.face_service = service
    application.state.performance_tracker = performance_tracker

    @application.middleware("http")
    async def record_request_performance(
        request: Request,
        call_next,
    ):
        started_at = perf_counter()
        response = await call_next(request)
        duration_ms = (perf_counter() - started_at) * 1000
        performance_tracker.record("http_request_ms", duration_ms)
        path_metric = (
            f"http_{request.method.lower()}_"
            f"{request.url.path.strip('/').replace('/', '_') or 'root'}_ms"
        )
        performance_tracker.record(path_metric, duration_ms)
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.3f}"
        return response

    application.add_exception_handler(
        FaceServiceError, face_service_error_handler
    )
    application.add_exception_handler(HTTPException, http_error_handler)
    application.add_exception_handler(
        RequestValidationError, validation_error_handler
    )
    application.add_exception_handler(Exception, unexpected_error_handler)
    application.include_router(router)
    return application


app = create_app()
