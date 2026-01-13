from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict

from langchain_community.vectorstores import FAISS

from .database import SessionLocal, DBSession, DBDocument

@dataclass
class SessionData:
    session_id: str
    vectorstore: Optional[FAISS] = None

class SessionStore:
    def __init__(self):
        self._cache: Dict[str, SessionData] = {}

    def get(self, session_id: str) -> SessionData:
        """
        Get existing session or create a new one.
        Ensures persistence in MySQL.
        """
        if session_id in self._cache:
            return self._cache[session_id]

        # Ensure session exists in DB
        db = SessionLocal()
        try:
            db_obj = db.query(DBSession).filter(DBSession.session_id == session_id).first()
            if not db_obj:
                db_obj = DBSession(session_id=session_id)
                db.add(db_obj)
                db.commit()
        finally:
            db.close()

        # Try to load vectorstore from disk
        vectorstore = self._load_vectorstore(session_id)
        
        session = SessionData(session_id=session_id, vectorstore=vectorstore)
        self._cache[session_id] = session
        return session

    def _load_vectorstore(self, session_id: str) -> Optional[FAISS]:
        from .llm import build_embeddings
        
        # Determine path (mirroring the path logic used elsewhere)
        # Based on file listing: storage/<uuid>/faiss/index.faiss
        base_path = os.path.join("storage", session_id, "faiss")
        
        if os.path.exists(base_path) and os.path.exists(os.path.join(base_path, "index.faiss")):
            try:
                embeddings = build_embeddings()
                return FAISS.load_local(base_path, embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                print(f"Error loading FAISS index for {session_id}: {e}")
                return None
        return None

    def save_vectorstore(self, session: SessionData):
        """Save the session's FAISS index to disk."""
        if not session.vectorstore:
            return
        
        base_path = os.path.join("storage", session.session_id, "faiss")
        os.makedirs(base_path, exist_ok=True)
        session.vectorstore.save_local(base_path)

    def add_document_record(self, session_id: str, filename: str, page_count: int, summary: Optional[str] = None, vector_ids: Optional[list[str]] = None):
        """Add a record of the uploaded document to MySQL."""
        import json
        db = SessionLocal()
        try:
            doc = DBDocument(
                session_id=session_id,
                filename=filename,
                page_count=page_count,
                summary=summary,
                vector_ids=json.dumps(vector_ids) if vector_ids else None
            )
            db.add(doc)
            db.commit()
        except Exception as e:
            print(f"Error saving document record: {e}")
            db.rollback()
        finally:
            db.close()

    def remove_document_record(self, session_id: str, filename: str):
        """Remove a document record from MySQL."""
        db = SessionLocal()
        try:
            db.query(DBDocument).filter(
                DBDocument.session_id == session_id,
                DBDocument.filename == filename
            ).delete()
            db.commit()
        except Exception as e:
            print(f"Error removing document record: {e}")
            db.rollback()
        finally:
            db.close()

SESSION_STORE = SessionStore()
