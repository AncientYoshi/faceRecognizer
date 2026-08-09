"""Tests for local embedding persistence."""

import sqlite3
import struct
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import psycopg
import pytest

from app.repositories.embedding_repository import (
    EmbeddingRepositoryError,
    PostgreSQLEmbeddingRepository,
    SQLiteEmbeddingRepository,
)


def embedding(value_index: int) -> tuple[float, ...]:
    values = [0.0] * 512
    values[value_index] = 1.0
    return tuple(values)


def test_sqlite_repository_persists_an_embedding(tmp_path: Path) -> None:
    database_path = tmp_path / "faces.db"
    repository = SQLiteEmbeddingRepository(str(database_path))
    repository.initialize()

    embedding_id = repository.upsert("STU-001", embedding(0))
    reopened_repository = SQLiteEmbeddingRepository(str(database_path))
    stored = reopened_repository.find_by_student_id("STU-001")

    assert stored is not None
    assert stored.embedding_id == embedding_id
    assert stored.student_id == "STU-001"
    assert stored.values == pytest.approx(embedding(0))


def test_sqlite_repository_atomically_replaces_a_students_embedding(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "faces.db"
    repository = SQLiteEmbeddingRepository(str(database_path))
    repository.initialize()

    first_id = repository.upsert("STU-001", embedding(0))
    replacement_id = repository.upsert("STU-001", embedding(1))
    stored = repository.find_by_student_id("STU-001")

    assert stored is not None
    assert replacement_id != first_id
    assert stored.embedding_id == replacement_id
    assert stored.values == pytest.approx(embedding(1))

    with sqlite3.connect(database_path) as connection:
        row_count = connection.execute(
            "SELECT COUNT(*) FROM face_embeddings"
        ).fetchone()[0]
    assert row_count == 1


def test_sqlite_repository_returns_none_for_an_unknown_student(
    tmp_path: Path,
) -> None:
    repository = SQLiteEmbeddingRepository(str(tmp_path / "faces.db"))
    repository.initialize()

    assert repository.find_by_student_id("UNKNOWN") is None


def test_sqlite_repository_loads_only_candidate_embeddings(
    tmp_path: Path,
) -> None:
    repository = SQLiteEmbeddingRepository(str(tmp_path / "faces.db"))
    repository.initialize()
    repository.upsert("STU-001", embedding(0))
    repository.upsert("STU-002", embedding(1))
    repository.upsert("STU-003", embedding(2))

    stored = repository.find_by_student_ids(("STU-003", "STU-001"))

    assert {record.student_id for record in stored} == {
        "STU-001",
        "STU-003",
    }


def postgres_repository() -> tuple[
    PostgreSQLEmbeddingRepository,
    MagicMock,
    MagicMock,
]:
    connection_factory = MagicMock()
    connection = connection_factory.return_value.__enter__.return_value
    repository = PostgreSQLEmbeddingRepository(
        "postgresql://user:password@database:5432/smart_attendance",
        timeout_seconds=4.2,
        connection_factory=connection_factory,
    )
    return repository, connection_factory, connection


def test_postgres_repository_validates_the_flyway_table() -> None:
    repository, connection_factory, connection = postgres_repository()

    repository.initialize()

    connection_factory.assert_called_once_with(
        "postgresql://user:password@database:5432/smart_attendance",
        connect_timeout=5,
    )
    assert "FROM face_embeddings" in connection.execute.call_args.args[0]


def test_postgres_repository_upserts_an_embedding() -> None:
    repository, _, connection = postgres_repository()
    connection.execute.return_value.fetchone.return_value = ("emb-123",)

    embedding_id = repository.upsert(
        "2f52f06f-59ed-4519-bb86-69cb59fb3197",
        embedding(0),
    )

    assert embedding_id == "emb-123"
    sql, parameters = connection.execute.call_args.args
    assert "ON CONFLICT (student_id) DO UPDATE" in sql
    assert parameters[0] == "2f52f06f-59ed-4519-bb86-69cb59fb3197"
    assert isinstance(parameters[2], bytes)
    assert len(parameters[2]) == 512 * 4
    assert parameters[3] == 512


def test_postgres_repository_loads_an_embedding() -> None:
    repository, _, connection = postgres_repository()
    timestamp = datetime(2026, 8, 2, tzinfo=UTC)
    connection.execute.return_value.fetchone.return_value = (
        "2f52f06f-59ed-4519-bb86-69cb59fb3197",
        "emb-123",
        struct.pack("<512f", *embedding(1)),
        512,
        timestamp,
        timestamp,
    )

    stored = repository.find_by_student_id(
        "2f52f06f-59ed-4519-bb86-69cb59fb3197"
    )

    assert stored is not None
    assert stored.embedding_id == "emb-123"
    assert stored.values == pytest.approx(embedding(1))
    assert stored.created_at == "2026-08-02T00:00:00+00:00"


def test_postgres_repository_loads_candidate_embeddings_in_one_query() -> None:
    repository, _, connection = postgres_repository()
    timestamp = datetime(2026, 8, 2, tzinfo=UTC)
    candidate_ids = (
        "2f52f06f-59ed-4519-bb86-69cb59fb3197",
        "12807f44-e4e2-464a-b525-9812b3dc0f3c",
    )
    connection.execute.return_value.fetchall.return_value = [
        (
            candidate_ids[1],
            "emb-456",
            struct.pack("<512f", *embedding(1)),
            512,
            timestamp,
            timestamp,
        )
    ]

    stored = repository.find_by_student_ids(candidate_ids)

    assert [record.student_id for record in stored] == [candidate_ids[1]]
    sql, parameters = connection.execute.call_args.args
    assert "student_id = ANY(%s::uuid[])" in sql
    assert parameters == (list(candidate_ids),)


def test_postgres_repository_wraps_connection_errors() -> None:
    connection_factory = MagicMock(
        side_effect=psycopg.OperationalError("database unavailable")
    )
    repository = PostgreSQLEmbeddingRepository(
        "postgresql://database/smart_attendance",
        connection_factory=connection_factory,
    )

    with pytest.raises(EmbeddingRepositoryError, match="Flyway migration"):
        repository.initialize()


def test_postgres_repository_rejects_invalid_embedding_dimension() -> None:
    repository, _, _ = postgres_repository()

    with pytest.raises(EmbeddingRepositoryError, match="511 values"):
        repository.upsert("student-id", tuple(0.0 for _ in range(511)))
