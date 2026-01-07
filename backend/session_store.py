from __future__ import annotations

import json
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Sequence
from urllib.parse import urlparse

import mysql.connector
from langchain_community.vectorstores import FAISS

from .llm import build_embeddings


@dataclass
class SessionData:
    session_id: str
    vectorstore: FAISS | None = None
    files: List[str] = field(default_factory=list)
    page_summaries: Dict[str, Dict[int, str]] = field(default_factory=lambda: defaultdict(dict))
    file_contents: Dict[str, bytes] = field(default_factory=dict)
    image_files: Dict[str, bytes] = field(default_factory=dict)
    file_compliance: Dict[str, Dict] = field(default_factory=dict)
    file_analyses: Dict[str, Dict] = field(default_factory=dict)
    questionnaire: List[Dict[str, Any]] = field(default_factory=list)
    rag_answers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    readme: str = ""


class SessionStore:
    def __init__(
        self,
        db_path: str = "sessions.db",
        storage_dir: str = "storage",
        db_url: str | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, SessionData] = {}

        self._session_db_url = db_url or os.getenv("SESSION_DB_URL")
        self._use_mysql = bool(self._session_db_url)
        self._mysql_config: Dict[str, Any] | None = None
        self.db_path = db_path

        if self._use_mysql:
            self._mysql_config = self._parse_mysql_url(self._session_db_url)

        self._init_db()

    def _parse_mysql_url(self, url: str) -> Dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme != "mysql":
            raise ValueError("SESSION_DB_URL must start with mysql://")

        database = parsed.path.lstrip("/")
        if not database:
            raise ValueError("SESSION_DB_URL must include a database name")

        return {
            "host": parsed.hostname or "127.0.0.1",
            "port": parsed.port or 3306,
            "user": parsed.username or "root",
            "password": parsed.password or "",
            "database": database,
        }

    def _connect_mysql(self) -> mysql.connector.connection.MySQLConnection:
        config = self._mysql_config
        if config is None:
            raise RuntimeError("MySQL configuration is missing")
        return mysql.connector.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"],
            autocommit=True,
            use_pure=True,
        )

    def _execute_sql(
        self,
        query: str,
        params: Sequence[Any] | None = None,
        *,
        fetchone: bool = False,
        fetchall: bool = False,
    ) -> Any:
        params = tuple(params or ())
        if self._use_mysql:
            conn = self._connect_mysql()
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                if fetchone:
                    return cursor.fetchone()
                if fetchall:
                    return cursor.fetchall()
            finally:
                cursor.close()
                conn.close()
            return None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetchone:
                return cursor.fetchone()
            if fetchall:
                return cursor.fetchall()
            return None

    def save_rag_answers(self, session_id: str, answers: Sequence[Dict[str, Any]]) -> None:
        if not answers:
            return

        if self._use_mysql:
            query = """
                INSERT INTO rag_answers (session_id, question, answer, extracted_answer, summary, sources)
                VALUES (%s, %s, %s, %s, %s, CAST(%s AS JSON))
            """
        else:
            query = """
                INSERT INTO rag_answers (session_id, question, answer, extracted_answer, summary, sources)
                VALUES (?, ?, ?, ?, ?, ?)
            """

        for item in answers:
            question = item.get("question", "")
            answer_text = item.get("answer", "")
            extracted = item.get("extracted_answer", "")
            summary = item.get("summary")
            sources = item.get("sources") or []
            sources_json = json.dumps(sources, ensure_ascii=False)
            params = (session_id, question, answer_text, extracted, summary, sources_json)
            self._execute_sql(query, params)

    def _init_db(self):
        if self._use_mysql:
            session_table = """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(255) PRIMARY KEY,
                    metadata JSON NOT NULL
                )
            """
            rag_table = """
                CREATE TABLE IF NOT EXISTS rag_answers (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    extracted_answer TEXT,
                    summary TEXT,
                    sources JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session (session_id)
                )
            """
        else:
            session_table = """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    metadata TEXT
                )
            """
            rag_table = """
                CREATE TABLE IF NOT EXISTS rag_answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    extracted_answer TEXT,
                    summary TEXT,
                    sources TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        self._execute_sql(session_table)
        self._execute_sql(rag_table)

        # Migration: Add extracted_answer column if it doesn't exist
        try:
            if self._use_mysql:
                # Check if column exists in MySQL
                check_col = """
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_name = 'rag_answers' 
                    AND column_name = 'extracted_answer'
                    AND table_schema = DATABASE()
                """
                res = self._execute_sql(check_col, fetchone=True)
                if res and res[0] == 0:
                    print("Migrating MySQL: Adding extracted_answer column...")
                    self._execute_sql("ALTER TABLE rag_answers ADD COLUMN extracted_answer TEXT AFTER answer")
            else:
                # Check if column exists in SQLite
                cols = self._execute_sql("PRAGMA table_info(rag_answers)", fetchall=True)
                col_names = [c[1] for c in cols]
                if "extracted_answer" not in col_names:
                    print("Migrating SQLite: Adding extracted_answer column...")
                    self._execute_sql("ALTER TABLE rag_answers ADD COLUMN extracted_answer TEXT")
        except Exception as e:
            print(f"Migration warning (can be ignored if column exists): {e}")

    def _get_storage_path(self, session_id: str) -> Path:
        path = self.storage_dir / session_id
        path.mkdir(exist_ok=True)
        return path

    def save(self, session_id: str):
        if session_id not in self._cache:
            return

        sess = self._cache[session_id]
        storage_path = self._get_storage_path(session_id)

        # Save FAISS
        if sess.vectorstore:
            sess.vectorstore.save_local(str(storage_path / "faiss"))

        # Save large binary files to disk
        files_dir = storage_path / "files"
        files_dir.mkdir(exist_ok=True)
        for fname, content in sess.file_contents.items():
            (files_dir / fname).write_bytes(content)
        
        images_dir = storage_path / "images"
        images_dir.mkdir(exist_ok=True)
        for fname, content in sess.image_files.items():
            (images_dir / fname).write_bytes(content)

        metadata = {
            "files": sess.files,
            "page_summaries": sess.page_summaries,
            "file_compliance": sess.file_compliance,
            "file_analyses": sess.file_analyses,
            "questionnaire": sess.questionnaire,
            "rag_answers": sess.rag_answers,
            "readme": sess.readme,
        }
        serialized = json.dumps(metadata, ensure_ascii=False)

        if self._use_mysql:
            query = """
                INSERT INTO sessions (session_id, metadata)
                VALUES (%s, CAST(%s AS JSON))
                ON DUPLICATE KEY UPDATE metadata = VALUES(metadata)
            """
        else:
            query = "INSERT OR REPLACE INTO sessions (session_id, metadata) VALUES (?, ?)"

        self._execute_sql(query, (session_id, serialized))

    def get(self, session_id: str) -> SessionData:
        if session_id in self._cache:
            return self._cache[session_id]

        if self._use_mysql:
            query = "SELECT metadata FROM sessions WHERE session_id = %s"
        else:
            query = "SELECT metadata FROM sessions WHERE session_id = ?"

        row = self._execute_sql(query, (session_id,), fetchone=True)

        if row:
            metadata_json = row[0]
            metadata = json.loads(metadata_json)
            sess = SessionData(
                session_id=session_id,
                files=metadata.get("files", []),
                page_summaries=metadata.get("page_summaries", {}),
                file_compliance=metadata.get("file_compliance", {}),
                file_analyses=metadata.get("file_analyses", {}),
                questionnaire=metadata.get("questionnaire", []),
                rag_answers=metadata.get("rag_answers", {}),
                readme=metadata.get("readme", ""),
            )

            storage_path = self._get_storage_path(session_id)

            faiss_path = storage_path / "faiss"
            if (faiss_path / "index.faiss").exists():
                sess.vectorstore = FAISS.load_local(
                    str(faiss_path),
                    build_embeddings(),
                    allow_dangerous_deserialization=True,
                )

            files_dir = storage_path / "files"
            if files_dir.exists():
                for f in files_dir.iterdir():
                    if f.is_file():
                        sess.file_contents[f.name] = f.read_bytes()

            images_dir = storage_path / "images"
            if images_dir.exists():
                for f in images_dir.iterdir():
                    if f.is_file():
                        sess.image_files[f.name] = f.read_bytes()

            self._cache[session_id] = sess
            return sess

        sess = SessionData(session_id=session_id)
        self._cache[session_id] = sess
        return sess

    def list_sessions(self) -> List[str]:
        try:
            query = "SELECT session_id FROM sessions"
            rows = self._execute_sql(query, fetchall=True)
            db_sessions = [row[0] for row in rows] if rows else []
        except Exception:
            db_sessions = []

        all_sessions = set(db_sessions) | set(self._cache.keys())
        return sorted(list(all_sessions))


SESSION_STORE = SessionStore()
