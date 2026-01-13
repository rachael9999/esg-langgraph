from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi import Body, HTTPException
import requests
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document

from .llm import summarize_page, call_text_llm
from .models import (
    AnswerItem,
    AnswerSource,
    QuestionRequest,
)
from .rag import add_documents, is_numeric_question, run_rag_with_depth, delete_documents_by_filename
from .session_store import SESSION_STORE
from .database import init_db
from .utils import (
    detect_signature_or_seal,
    detect_signature_in_pdf,
    is_supported,
    parse_document,
    get_pdf_page_count,
    detect_signature_in_image,
    detect_signature_with_vl_model,
)

app = FastAPI(title="ESG LangGraph Service")

@app.on_event("startup")
def on_startup():
    init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/session/{session_id}/upload-and-index")
async def upload_and_index(
    session_id: str,
    file: UploadFile = File(...),
) -> dict:
    """
    Step 1: Save the file.
    Step 2: Parse and summarize pages.
    Step 3: Index into FAISS immediately.
    """
    content = await file.read()
    filename = str(file.filename or "unknown")

    if not is_supported(filename):
        raise HTTPException(status_code=400, detail="文件格式不支持。")

    # SESSION_STORE.get automatically creates a new session if it doesn't exist
    session = SESSION_STORE.get(session_id)

    try:
        # Parse and summarize (do not persist file bytes or summaries in session)
        parsed = parse_document(filename, content)

        docs = []
        for page in parsed.pages:
            # Generate a quick summary for metadata
            summary = summarize_page(page.text)
            docs.append(
                Document(
                    page_content=page.text,
                    metadata={
                        "filename": filename,
                        "page_number": page.page_number,
                        "summary": summary,
                    },
                )
            )

        # Add to FAISS (session holds only vectorstore/faiss metadata)
        add_documents(session, docs)

        # Collect vector IDs from docs metadata
        vector_ids = [doc.metadata.get("vector_id") for doc in docs if doc.metadata and doc.metadata.get("vector_id")]

        SESSION_STORE.save_vectorstore(session)
        SESSION_STORE.add_document_record(
            session_id, 
            filename, 
            len(parsed.pages), 
            docs[0].metadata.get("summary") if docs else None,
            vector_ids
        )

        return {
            "status": "success",
            "filename": filename,
            "session_id": session_id,
            "page_count": len(parsed.pages),
            "message": "文件已成功上传并建立索引。"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"上传或索引失败: {str(e)}")


@app.delete("/session/{session_id}/remove-document")
async def remove_document(session_id: str, filename: str = Form(...)):
    """Remove a document from the session by filename."""
    session = SESSION_STORE.get(session_id)
    
    try:
        delete_documents_by_filename(session, filename)
        return {
            "status": "success",
            "session_id": session_id,
            "filename": filename,
            "message": "文档已从向量存储中软删除。"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


def best_source(docs: List[Document]) -> AnswerSource | None:
    if not docs:
        return None
    metadata = docs[0].metadata
    return AnswerSource(
        filename=metadata.get("filename", "unknown"),
        page_number=int(metadata.get("page_number", 1)),
    )

@app.post("/ask")
async def ask_questions(payload: QuestionRequest) -> dict:
    session = SESSION_STORE.get(payload.session_id)
    answers: List[AnswerItem] = []

    for question in payload.questions:
        rag_state = run_rag_with_depth(question, session, depth=payload.depth)
        sources: List[AnswerSource] = []
        summary: str | None = None
        retrieved_docs = rag_state.get("retrieved_docs")
        if retrieved_docs:
            for doc in retrieved_docs:
                metadata = doc.metadata or {}
                filename = metadata.get("filename", "unknown")
                try:
                    page_num = int(metadata.get("page_number", 1))
                except Exception:
                    page_num = 1
                # prefer summary from document metadata
                summary_text = metadata.get("summary")
                sources.append(
                    AnswerSource(
                        filename=filename,
                        page_number=page_num,
                        summary=summary_text,
                    )
                )
            # Combine summaries from all sources instead of just the first one
            all_summaries = []
            for src in sources:
                if src.summary:
                    all_summaries.append(f"[{src.filename} 第{src.page_number}页]: {src.summary}")
            summary = "\n".join(all_summaries) if all_summaries else None

        answer_text = rag_state.get("answer") or "未找到相关内容。"
        if is_numeric_question(question) and retrieved_docs:
            # Try to find the most relevant page mentioned in the text answer
            target_page = None
            target_filename = None
            
            # Heuristic: look for "第 X 页" or "第X页" in the answer text
            page_matches = re.findall(r"第\s*(\d+)\s*页", answer_text)
            if page_matches:
                target_page = int(page_matches[0])
                # Find which filename this page belongs to from retrieved_docs
                for doc in retrieved_docs:
                    if doc.metadata.get("page_number") == target_page:
                        target_filename = doc.metadata.get("filename")
                        break
            
            # Fallback to best_source if no page mentioned or not found in docs
            if not target_page or not target_filename:
                primary_source = best_source(retrieved_docs)
                if primary_source:
                    target_page = primary_source.page_number
                    target_filename = primary_source.filename


        answers.append(
            AnswerItem(
                question=question,
                answer=answer_text,
                extracted_answer=rag_state.get("extracted_answer"),
                sources=sources,
                summary=summary,
            )
        )
    return {
        "session_id": payload.session_id,
        "answerItems": [ans.dict() for ans in answers],
    }

@app.post("/compliance/detect-seal")
async def detect_signature_uploaded_file(file: UploadFile = File(...)) -> dict:
    """Explicitly detect signature/seal in an uploaded file."""
    content = await file.read()
    filename = str(file.filename or "uploaded_file")
    suffix = Path(filename).suffix.lower()
    
    try:
        signature = False
        signature_note = None
        
        if suffix == ".pdf":
            # We need page count, but avoid full text parsing
            num_pages = get_pdf_page_count(content)
            signature, signature_note = detect_signature_in_pdf(content, num_pages)
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            signature, signature_note = detect_signature_in_image(content)
        else:
            # Fallback for docx/txt
            parsed = parse_document(filename, content)
            full_text = "\n".join(page.text for page in parsed.pages)
            signature = detect_signature_or_seal(full_text)
            signature_note = "Text-based detection" if signature else "No keywords found"

        return {
            "filename": filename,
            "has_signature_or_seal": signature,
            "signature_note": signature_note
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/compliance/detect-signature-vl")
async def detect_signature_vl_uploaded_file(file: UploadFile = File(...)) -> dict:
    """Detect handwritten signature in an uploaded file (PDF or image) using VL model."""
    content = await file.read()
    filename = str(file.filename or "uploaded_file")
    suffix = Path(filename).suffix.lower()
    
    try:
        signature = False
        signature_note = None
        
        if suffix == ".pdf":
            num_pages = get_pdf_page_count(content)
            signature, signature_note = detect_signature_with_vl_model(content, num_pages)
        elif suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
            # For images, render to base64 and call VL model directly
            img_b64 = base64.b64encode(content).decode()
            prompt = "这张图片中是否有手写签名？请回答是或否，并简要说明。"
            from .llm import call_vl_llm
            response = call_vl_llm(img_b64, prompt)
            if "是" in response.lower() or "yes" in response.lower():
                signature = True
                signature_note = "检测到手写签名。"
            else:
                signature_note = "未检测到手写签名。"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF and image files are supported.")
        
        return {
            "filename": filename,
            "has_signature": signature,
            "signature_note": signature_note
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")