"""Tests for the one-time SQLite-to-PostgreSQL migration helper."""

import sqlite3
import struct
from pathlib import Path

from scripts.migrate_sqlite_embeddings_to_postgres import read_sqlite_records


def test_reads_existing_sqlite_embedding_records(tmp_path: Path) -> None:
    database_path = tmp_path / "faces.db"
    values = [0.0] * 512
    values[7] = 1.0
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE face_embeddings (
                student_id TEXT PRIMARY KEY,
                embedding_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO face_embeddings VALUES (?, ?, ?, ?, ?, ?)",
            (
                "2f52f06f-59ed-4519-bb86-69cb59fb3197",
                "emb-123",
                sqlite3.Binary(struct.pack("<512f", *values)),
                512,
                "2026-08-01T00:00:00+00:00",
                "2026-08-02T00:00:00+00:00",
            ),
        )

    records = read_sqlite_records(database_path)

    assert len(records) == 1
    assert records[0][0] == "2f52f06f-59ed-4519-bb86-69cb59fb3197"
    assert records[0][1] == "emb-123"
    unpacked = struct.unpack("<512f", records[0][2])
    assert unpacked[7] == 1.0
    assert len(records[0][2]) == 512 * 4


def test_skips_legacy_non_uuid_student_ids(tmp_path: Path) -> None:
    database_path = tmp_path / "faces.db"
    values = [0.0] * 512
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE face_embeddings (
                student_id TEXT PRIMARY KEY,
                embedding_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO face_embeddings VALUES (?, ?, ?, ?, ?, ?)",
            (
                "STU001",
                "emb-legacy",
                sqlite3.Binary(struct.pack("<512f", *values)),
                512,
                "2026-08-01T00:00:00+00:00",
                "2026-08-02T00:00:00+00:00",
            ),
        )

    assert read_sqlite_records(database_path) == []
