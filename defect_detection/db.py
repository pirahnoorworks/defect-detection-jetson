from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Tuple


def init_db(db_path: str | Path | None = None) -> Path:
    db_file = Path(db_path or "defect_inspections.db")
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()
    return db_file


def save_inspection(db_path: str | Path | None, image_name: str, label: str, confidence: float) -> None:
    db_file = init_db(db_path)
    conn = sqlite3.connect(db_file)
    conn.execute(
        "INSERT INTO inspections (image_name, label, confidence) VALUES (?, ?, ?)",
        (image_name, label, confidence),
    )
    conn.commit()
    conn.close()


def list_inspections(db_path: str | Path | None = None) -> List[Tuple[int, str, str, float, str]]:
    db_file = init_db(db_path)
    conn = sqlite3.connect(db_file)
    rows = conn.execute(
        "SELECT id, image_name, label, confidence, created_at FROM inspections ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return rows
