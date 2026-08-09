"""Public API response schemas."""

from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class HealthResponse(ApiModel):
    status: str = "UP"


class RegisterFaceResponse(ApiModel):
    success: bool
    embedding_id: str = Field(alias="embeddingId")


class VerifyFaceResponse(ApiModel):
    matched: bool
    similarity: float


class IdentifyFaceResponse(ApiModel):
    matched: bool
    student_id: UUID | None = Field(alias="studentId")
    similarity: Annotated[float, Field(ge=0.0, le=1.0)]
    liveness_passed: bool = Field(alias="livenessPassed")
    reason: Literal[
        "MATCHED",
        "NOT_MATCHED",
        "NO_FACE_DETECTED",
        "MULTIPLE_FACES",
        "LIVENESS_FAILED",
    ]

    @model_validator(mode="after")
    def validate_match_identity(self) -> Self:
        if self.matched and self.student_id is None:
            raise ValueError("A matched response requires studentId.")
        if self.matched and self.reason != "MATCHED":
            raise ValueError("A matched response requires reason MATCHED.")
        if not self.matched and self.student_id is not None:
            raise ValueError("An unmatched response must not include studentId.")
        return self


class BoundingBox(ApiModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectFaceResponse(ApiModel):
    face_found: bool = Field(alias="faceFound")
    bounding_box: BoundingBox = Field(alias="boundingBox")
    confidence: float


class EmbeddingResponse(ApiModel):
    dimension: Literal[512] = 512
    embedding: Annotated[list[float], Field(min_length=512, max_length=512)]


class MetricSummaryResponse(ApiModel):
    count: int
    minimum_ms: float = Field(alias="minimumMs")
    maximum_ms: float = Field(alias="maximumMs")
    average_ms: float = Field(alias="averageMs")
    p50_ms: float = Field(alias="p50Ms")
    p95_ms: float = Field(alias="p95Ms")
    p99_ms: float = Field(alias="p99Ms")


class ProcessMetricsResponse(ApiModel):
    resident_memory_bytes: int = Field(alias="residentMemoryBytes")
    virtual_memory_bytes: int = Field(alias="virtualMemoryBytes")
    memory_percent: float = Field(alias="memoryPercent")
    cpu_percent: float = Field(alias="cpuPercent")
    system_cpu_percent: float = Field(alias="systemCpuPercent")
    thread_count: int = Field(alias="threadCount")


class PerformanceResponse(ApiModel):
    captured_at: str = Field(alias="capturedAt")
    uptime_seconds: float = Field(alias="uptimeSeconds")
    sample_limit_per_metric: int = Field(alias="sampleLimitPerMetric")
    metrics: dict[str, MetricSummaryResponse]
    process: ProcessMetricsResponse


class ErrorDetail(ApiModel):
    code: str
    message: str


class ErrorResponse(ApiModel):
    error: ErrorDetail
