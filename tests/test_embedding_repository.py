"""Tests for local embedding persistence."""

import sqlite3
from pathlib import Path

import pytest

from app.repositories.embedding_repository import SQLiteEmbeddingRepository


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
