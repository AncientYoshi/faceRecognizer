"""Environment-based application configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    app_name: str = "AI Face Service"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    max_upload_size_mb: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    max_identify_candidates: int = Field(default=500, ge=1, le=10_000)
    face_database_url: str | None = None
    face_database_path: str = "data/faces.db"
    face_database_timeout_seconds: float = Field(default=5.0, gt=0.0, le=60.0)

    insightface_model_name: str = "buffalo_l"
    insightface_model_root: str = "~/.insightface"
    insightface_providers: str = "CPUExecutionProvider"
    insightface_context_id: int = 0
    insightface_detection_threshold: float = Field(
        default=0.50, ge=0.0, le=1.0
    )
    insightface_detection_width: int = Field(default=640, ge=160)
    insightface_detection_height: int = Field(default=640, ge=160)
    max_image_pixels: int = Field(default=20_000_000, ge=1_000_000)
    min_image_width: int = Field(default=160, ge=1)
    min_image_height: int = Field(default=160, ge=1)
    min_face_size_pixels: int = Field(default=80, ge=1)
    min_blur_score: float = Field(default=30.0, ge=0.0)
    performance_sample_limit: int = Field(default=1000, ge=10, le=100_000)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def insightface_provider_list(self) -> list[str]:
        return [
            provider.strip()
            for provider in self.insightface_providers.split(",")
            if provider.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
