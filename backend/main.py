from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi import Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document

from .llm import summarize_page, call_text_llm, call_vl_llm, QwenConfigError
from .models import (
    AnswerItem,
    AnswerSource,
    ComplianceResult,
    DocumentIngestion,
    PageSummary,
    QuestionnaireResponse,
    QuestionRequest,
    QuestionResponse,
    ReadmeRequest,
)
from .rag import add_documents, is_numeric_question, run_rag_with_depth
from .session_store import SESSION_STORE
from .utils import (
    detect_company_name,
    detect_signature_or_seal,
    is_supported,
    parse_document,
    parse_questionnaire,
    render_pdf_page_to_jpg,
)

app = FastAPI(title="ESG LangGraph Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/sessions")
async def list_sessions() -> List[str]:
    return SESSION_STORE.list_sessions()


@app.post("/upload", response_model=DocumentIngestion)
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
) -> DocumentIngestion:
    """Accept file upload and schedule heavy parsing/indexing in background.

    This endpoint returns quickly with a minimal `DocumentIngestion` response while
    the real parsing, summarization and vector indexing happen asynchronously.
    """
    content = await file.read()
    # quick format check
    if not is_supported(file.filename):
        compliance = ComplianceResult(
            is_supported=False,
            notes=["文件格式不支持。"],
        )
        return DocumentIngestion(
            filename=file.filename,
            page_summaries=[],
            compliance=compliance,
        )

    # persist raw bytes immediately so subsequent requests can reference them
    session = SESSION_STORE.get(session_id)
    if file.filename not in session.files:
        session.files.append(file.filename)
    session.file_contents[file.filename] = content
    SESSION_STORE.save(session_id)

    # schedule background processing
    def _process_upload(sid: str, fname: str, data: bytes) -> None:
        try:
            parsed = parse_document(fname, data)
            full_text = "\n".join(page.text for page in parsed.pages)
            company_name = detect_company_name(full_text)
            signature = detect_signature_or_seal(full_text)

            # If company name not found by regex, ask LLM to extract it (background only)
            if not company_name:
                try:
                    prompt = (
                        "请从下面的文本中提取公司全称，如果无法提取则仅返回空字符串。" + "\n\n" + full_text[:4000]
                    )
                    inferred = call_text_llm(prompt)
                    inferred = inferred.strip()
                    # simple heuristics: if non-empty and contains common company suffix, accept
                    if inferred and re.search(r"(公司|集团|有限责任公司|股份有限公司)", inferred):
                        company_name = inferred
                except Exception:
                    pass

            # If signature not detected, ask LLM to check for signature or seal
            if not signature:
                try:
                    prompt2 = (
                        "下面的文本是否包含签名、签章或公司盖章的描述？如果包含请回答 'yes'，否则回答 'no'。" + "\n\n" + full_text[:2000]
                    )
                    resp = call_text_llm(prompt2).lower()
                    if "yes" in resp or "是" in resp:
                        signature = True
                except Exception:
                    pass

            compliance = ComplianceResult(
                is_supported=True,
                company_name=company_name,
                company_name_ok=bool(company_name),
                has_signature_or_seal=signature,
                notes=[],
            )

            # persist compliance into session store for retrieval/debugging
            SESSION_STORE.get(sid).file_compliance[fname] = {
                "is_supported": compliance.is_supported,
                "company_name": compliance.company_name,
                "company_name_ok": compliance.company_name_ok,
                "has_signature_or_seal": compliance.has_signature_or_seal,
                "notes": compliance.notes,
            }

            sess = SESSION_STORE.get(sid)
            page_summaries: List[PageSummary] = []
            docs: List[Document] = []
            for page in parsed.pages:
                summary = summarize_page(page.text)
                page_summaries.append(PageSummary(page_number=page.page_number, summary=summary))
                sess.page_summaries[parsed.filename][page.page_number] = summary

                docs.append(
                    Document(
                        page_content=page.text,
                        metadata={
                            "filename": parsed.filename,
                            "page_number": page.page_number,
                            "summary": summary,
                        },
                    )
                )
            # attempt to add documents to vectorstore; on failure, log and continue
            try:
                add_documents(sess, docs)
                SESSION_STORE.save(sid)
            except Exception:
                import traceback

                traceback.print_exc()
                # record a note so clients can see embedding/indexing failed
                sess.file_compliance[fname] = sess.file_compliance.get(fname, {})
                sess.file_compliance[fname]["notes"] = ["indexing_failed"]
        except Exception:
            # background errors should not crash the server; log to stdout for now
            import traceback

            traceback.print_exc()

    if background_tasks is not None:
        background_tasks.add_task(_process_upload, session_id, file.filename, content)
    else:
        # fallback: run synchronously if BackgroundTasks not provided
        _process_upload(session_id, file.filename, content)

    compliance = ComplianceResult(is_supported=True, notes=["processing"])
    return DocumentIngestion(filename=file.filename, page_summaries=[], compliance=compliance)


def best_source(docs: List[Document]) -> AnswerSource | None:
    if not docs:
        return None
    metadata = docs[0].metadata
    return AnswerSource(
        filename=metadata.get("filename", "unknown"),
        page_number=int(metadata.get("page_number", 1)),
    )


@app.post("/ask", response_model=QuestionResponse)
async def ask_questions(payload: QuestionRequest) -> QuestionResponse:
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
                # prefer summary from document metadata, fallback to session page_summaries
                summary_text = metadata.get("summary") or session.page_summaries.get(filename, {}).get(page_num)
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

            if target_filename and target_filename.lower().endswith(".pdf"):
                pdf_bytes = session.file_contents.get(target_filename)
                if pdf_bytes:
                    image_path = Path("/tmp") / f"{uuid.uuid4()}-page-{target_page}.jpg"
                    render_pdf_page_to_jpg(pdf_bytes, target_page, image_path)
                    try:
                        vl_answer = call_vl_llm(image_path, question)
                        # Only append if VL actually found something useful and doesn't contradict a strong text answer
                        if vl_answer and "未找到" not in vl_answer and "无法确定" not in vl_answer:
                            answer_text = f"{answer_text}\n\n(视觉核对结果 [第{target_page}页]: {vl_answer})"
                    except QwenConfigError:
                        pass

        answers.append(
            AnswerItem(
                question=question,
                answer=answer_text,
                extracted_answer=rag_state.get("extracted_answer"),
                sources=sources,
                summary=summary,
            )
        )

    stored_answers = []
    for ans in answers:
        stored_answers.append(
            {
                "question": ans.question,
                "answer": ans.answer,
                "extracted_answer": ans.extracted_answer,
                "summary": ans.summary,
                "sources": [
                    {
                        "filename": src.filename,
                        "page_number": src.page_number,
                        "summary": src.summary,
                    }
                    for src in ans.sources
                ],
            }
        )
    SESSION_STORE.save_rag_answers(payload.session_id, stored_answers)

    return QuestionResponse(session_id=payload.session_id, answers=answers)


@app.post("/parse-readme", response_model=QuestionnaireResponse)
async def parse_readme_endpoint(payload: ReadmeRequest) -> QuestionnaireResponse:
    items = parse_questionnaire(payload.content)
    # We don't have session_id here, but we can save it when the user actually uses it in a session
    return QuestionnaireResponse(items=items)


@app.get("/session/{session_id}/questionnaire")
def get_questionnaire(session_id: str) -> dict:
    session = SESSION_STORE.get(session_id)
    return {"items": session.questionnaire, "rag_answers": session.rag_answers}


@app.post("/session/{session_id}/questionnaire")
def save_questionnaire(session_id: str, payload: dict = Body(...)) -> dict:
    session = SESSION_STORE.get(session_id)
    session.questionnaire = payload.get("items", [])
    SESSION_STORE.save(session_id)
    return {"status": "ok"}


@app.get("/session/{session_id}/compliance")
def get_session_compliance(session_id: str) -> dict:
    """Return all stored `file_compliance` entries for a session."""
    session = SESSION_STORE.get(session_id)
    return session.file_compliance


@app.get("/session/{session_id}/compliance/{filename}")
def get_file_compliance(session_id: str, filename: str) -> dict:
    """Return compliance info for a single file (or 404-like empty dict)."""
    session = SESSION_STORE.get(session_id)
    return session.file_compliance.get(filename, {})


@app.get("/session/{session_id}/status")
def get_session_status(session_id: str) -> dict:
    """Return session status including whether a vectorstore exists for this session."""
    session = SESSION_STORE.get(session_id)
    return {
        "files": session.files,
        "image_files": list(session.image_files.keys()),
        "file_compliance": session.file_compliance,
        "has_vectorstore": session.vectorstore is not None,
    }


@app.put("/session/{session_id}/compliance/{filename}")
def update_file_compliance(session_id: str, filename: str, payload: dict = Body(...)) -> dict:
    """Update or set compliance info for a given session/file.

    Expects a JSON body with fields matching the compliance structure, e.g.:
    {"is_supported": true, "company_name": "...", "company_name_ok": true, "has_signature_or_seal": false, "notes": []}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be a JSON object")
    session = SESSION_STORE.get(session_id)
    # ensure there is a place
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    session.file_compliance[filename] = payload
    SESSION_STORE.save(session_id)
    return session.file_compliance[filename]
