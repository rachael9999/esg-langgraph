#!/usr/bin/env python3
"""Migrate sessions.db metadata to minimal Qdrant-only format.

This script creates a backup of `sessions.db` and updates every session's
`metadata` column to a JSON object containing only `qdrant_last_updated`.
It will try to infer `qdrant_last_updated` from the local Qdrant storage
directory modification time under `storage/<session_id>/qdrant`.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "sessions.db"
STORAGE_DIR = ROOT / "storage"


def infer_qdrant_mtime(session_id: str) -> Optional[str]:
    qdrant_dir = STORAGE_DIR / session_id / "qdrant"
    if qdrant_dir.exists():
        try:
            mtime = datetime.fromtimestamp(qdrant_dir.stat().st_mtime, tz=timezone.utc)
            return mtime.isoformat()
        except Exception:
            return None
    return None


def backup_db(path: Path) -> Path:
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    return bak


def migrate():
    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}")
        return

    bak = backup_db(DB_PATH)
    print(f"Backup created: {bak}")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    cur.execute("SELECT session_id, metadata FROM sessions")
    rows = cur.fetchall()
    total = len(rows)
    updated = 0
    examples = []

    for session_id, metadata_json in rows:
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            meta = {}

        qdrant_last = meta.get("qdrant_last_updated") or infer_qdrant_mtime(session_id)

        new_meta = {"qdrant_last_updated": qdrant_last}
        new_serial = json.dumps(new_meta, ensure_ascii=False)

        if new_serial != (metadata_json or ""):
            cur.execute("UPDATE sessions SET metadata = ? WHERE session_id = ?", (new_serial, session_id))
            updated += 1
            if len(examples) < 20:
                examples.append((session_id, new_meta))

    conn.commit()
    conn.close()

    print(f"Processed {total} sessions; updated {updated} sessions.")
    if examples:
        print("Examples of updated sessions:")
        for sid, nm in examples:
            print(f" - {sid}: {nm}")


if __name__ == "__main__":
    migrate()
