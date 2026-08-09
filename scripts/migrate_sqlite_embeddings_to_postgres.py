"""Copy existing SQLite face embeddings into the shared PostgreSQL database."""

from __future__ import annotations

import argparse
import os
import sqlite3
import struct
from pathlib import Path
from uuid import UUID

import psycopg

from app.services.face_service import EMBEDDING_DIMENSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the current SQLite biometric templates into PostgreSQL "
            "without changing their embedding IDs."
        )
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=Path("data/faces.db"),
        help="Path to the existing SQLite database.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("FACE_DATABASE_URL"),
        help="PostgreSQL URL; defaults to FACE_DATABASE_URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count records without writing PostgreSQL.",
    )
    return parser.parse_args()


def read_sqlite_records(
    database_path: Path,
) -> list[tuple[str, str, bytes, int, str, str]]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                student_id,
                embedding_id,
                embedding,
                dimension,
                created_at,
                updated_at
            FROM face_embeddings
            ORDER BY student_id
            """
        ).fetchall()

    records: list[tuple[str, str, bytes, int, str, str]] = []
    for row in rows:
        try:
            UUID(str(row[0]))
        except ValueError:
            print(
                f"Skipping legacy student ID {row[0]!r}; "
                "Spring student IDs must be UUIDs."
            )
            continue
        dimension = int(row[3])
        if dimension != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Student {row[0]!r} has dimension {dimension}, expected 512."
            )
        packed_values = bytes(row[2])
        struct.unpack(f"<{EMBEDDING_DIMENSION}f", packed_values)
        records.append(
            (
                str(row[0]),
                str(row[1]),
                packed_values,
                dimension,
                row[4],
                row[5],
            )
        )
    return records


def migrate(
    records: list[tuple[str, str, bytes, int, str, str]],
    database_url: str,
) -> int:
    migrated = 0
    with psycopg.connect(database_url) as connection:
        for record in records:
            cursor = connection.execute(
                """
                INSERT INTO face_embeddings (
                    student_id,
                    embedding_id,
                    embedding,
                    dimension,
                    created_at,
                    updated_at
                )
                SELECT %s, %s, %s, %s, %s, %s
                WHERE EXISTS (
                    SELECT 1 FROM students WHERE id = %s
                )
                ON CONFLICT (student_id) DO UPDATE SET
                    embedding_id = EXCLUDED.embedding_id,
                    embedding = EXCLUDED.embedding,
                    dimension = EXCLUDED.dimension,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at
                """,
                (*record, record[0]),
            )
            migrated += cursor.rowcount
    return migrated


def main() -> int:
    args = parse_args()
    database_path = args.sqlite.expanduser().resolve()
    if not database_path.is_file():
        raise SystemExit(f"SQLite database not found: {database_path}")

    records = read_sqlite_records(database_path)
    if args.dry_run:
        print(f"Validated {len(records)} SQLite embedding record(s).")
        return 0
    if not args.database_url:
        raise SystemExit(
            "Set FACE_DATABASE_URL or pass --database-url before migrating."
        )

    migrated = migrate(records, args.database_url)
    print(
        f"Migrated {migrated} of {len(records)} eligible embedding "
        "record(s) to PostgreSQL."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
