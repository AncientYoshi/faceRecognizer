"""Embedding repository contract and local SQLite implementation."""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.services.face_service import EMBEDDING_DIMENSION


@dataclass(frozen=True)
class StoredEmbedding:
    student_id: str
    embedding_id: str
    values: tuple[float, ...]
    created_at: str
    updated_at: str


class EmbeddingRepository(Protocol):
    def initialize(self) -> None:
        """Create or validate required storage structures."""

    def upsert(
        self,
        student_id: str,
        values: tuple[float, ...],
    ) -> str:
        """Insert or replace a student's embedding and return its new ID."""

    def find_by_student_id(self, student_id: str) -> StoredEmbedding | None:
        """Return the student's current embedding, if registered."""


class EmbeddingRepositoryError(RuntimeError):
    """Raised when embedding persistence fails."""


class SQLiteEmbeddingRepository:
    """One-current-embedding-per-student SQLite repository."""

    def __init__(
        self,
        database_path: str,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._database_path = str(Path(database_path).expanduser().resolve())
        self._timeout_seconds = timeout_seconds

    @property
    def database_path(self) -> str:
        return self._database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path,
            timeout=self._timeout_seconds,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(
            f"PRAGMA busy_timeout = {int(self._timeout_seconds * 1000)}"
        )
        return connection

    def initialize(self) -> None:
        try:
            Path(self._database_path).parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS face_embeddings (
                        student_id TEXT PRIMARY KEY,
                        embedding_id TEXT UNIQUE NOT NULL,
                        embedding BLOB NOT NULL,
                        dimension INTEGER NOT NULL CHECK (dimension = 512),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
        except (OSError, sqlite3.Error) as exc:
            raise EmbeddingRepositoryError(
                "Could not initialize the face embedding database."
            ) from exc

    def upsert(
        self,
        student_id: str,
        values: tuple[float, ...],
    ) -> str:
        if len(values) != EMBEDDING_DIMENSION:
            raise EmbeddingRepositoryError(
                f"Cannot store an embedding with {len(values)} values."
            )

        embedding_id = str(uuid4())
        timestamp = datetime.now(UTC).isoformat()
        try:
            packed_values = struct.pack(
                f"<{EMBEDDING_DIMENSION}f",
                *values,
            )
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO face_embeddings (
                        student_id,
                        embedding_id,
                        embedding,
                        dimension,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(student_id) DO UPDATE SET
                        embedding_id = excluded.embedding_id,
                        embedding = excluded.embedding,
                        dimension = excluded.dimension,
                        updated_at = excluded.updated_at
                    """,
                    (
                        student_id,
                        embedding_id,
                        sqlite3.Binary(packed_values),
                        EMBEDDING_DIMENSION,
                        timestamp,
                        timestamp,
                    ),
                )
        except (sqlite3.Error, struct.error) as exc:
            raise EmbeddingRepositoryError(
                "Could not store the face embedding."
            ) from exc

        return embedding_id

    def find_by_student_id(self, student_id: str) -> StoredEmbedding | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT
                        student_id,
                        embedding_id,
                        embedding,
                        dimension,
                        created_at,
                        updated_at
                    FROM face_embeddings
                    WHERE student_id = ?
                    """,
                    (student_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise EmbeddingRepositoryError(
                "Could not load the registered face embedding."
            ) from exc

        if row is None:
            return None
        if row["dimension"] != EMBEDDING_DIMENSION:
            raise EmbeddingRepositoryError(
                "The stored face embedding has an invalid dimension."
            )

        try:
            values = struct.unpack(
                f"<{EMBEDDING_DIMENSION}f",
                row["embedding"],
            )
        except struct.error as exc:
            raise EmbeddingRepositoryError(
                "The stored face embedding is corrupted."
            ) from exc

        return StoredEmbedding(
            student_id=row["student_id"],
            embedding_id=row["embedding_id"],
            values=values,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
