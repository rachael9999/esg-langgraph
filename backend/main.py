from __future__ import annotations

import re
import uuid
import json
from pathlib import Path
from typing import List, Optional

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
    detect_signature_in_pdf,
    detect_report_type,
    detect_industry,
    detect_company_size,
    detect_region,
    detect_target,
    is_supported,
    parse_document,
    parse_questionnaire,
    render_pdf_page_to_jpg,
    get_pdf_page_count,
    detect_signature_in_image,
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
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> DocumentIngestion:
    """Accept file upload and schedule heavy parsing/indexing in background.

    This endpoint returns quickly with a minimal `DocumentIngestion` response while
    the real parsing, summarization and vector indexing happen asynchronously.
    """
    content = await file.read()
    # quick format check
    if not is_supported(file.filename or ""):
        compliance = ComplianceResult(
            is_supported=False,
            notes=["文件格式不支持。"],
        )
        return DocumentIngestion(
            filename=str(file.filename or ""),
            page_summaries=[],
            compliance=compliance,
        )

    # persist raw bytes immediately so subsequent requests can reference them
    session = SESSION_STORE.get(session_id)
    if file.filename:
        if file.filename not in session.files:
            session.files.append(file.filename)
        session.file_contents[file.filename] = content
    SESSION_STORE.save(session_id)

    # schedule background processing
    def _process_upload(sid: str, fname: str, data: bytes) -> None:
        sess = SESSION_STORE.get(sid)
        try:
            parsed = parse_document(fname, data)
            full_text = "\n".join(page.text for page in parsed.pages)
            # Use LLM to extract company name and industry for all file types
            company_name = None
            industry = None
            try:
                prompt_ci = (
                    "请从下面的文本中提取公司全称和所属行业，并以JSON格式返回，示例：{\"company_name\": \"...\", \"industry\": \"...\"}。"
                    + "\n\n" + full_text[:10000]
                )
                resp = call_text_llm(prompt_ci)
                # try to parse JSON from response
                parsed_json = None
                try:
                    parsed_json = json.loads(resp)
                except Exception:
                    # attempt to find JSON substring
                    start = resp.find("{")
                    end = resp.rfind("}")
                    if start != -1 and end != -1:
                        try:
                            parsed_json = json.loads(resp[start:end+1])
                        except Exception:
                            parsed_json = None

                if isinstance(parsed_json, dict):
                    company_name = parsed_json.get("company_name") or parsed_json.get("company")
                    industry = parsed_json.get("industry")
            except Exception:
                company_name = None

            # fallback heuristics
            if not company_name:
                company_name = detect_company_name(full_text)
            if not industry:
                industry = detect_industry(full_text)

            # signature: prefer visual check for PDFs using VLM across all pages
            signature = False
            signature_note = None
            try:
                if fname.lower().endswith(".pdf"):
                    try:
                        signature, signature_note = detect_signature_in_pdf(data, len(parsed.pages))
                    except Exception:
                        signature = detect_signature_or_seal(full_text)
                        signature_note = None
                else:
                    # non-pdf fallback: use text-based detection
                    signature = detect_signature_or_seal(full_text)
                    signature_note = None
            except Exception:
                signature = detect_signature_or_seal(full_text)
                signature_note = None

            # Detect additional metadata
            company_size = detect_company_size(full_text)
            region = detect_region(full_text)
            target = detect_target(full_text)

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

            report_type = detect_report_type(full_text)

            compliance = ComplianceResult(
                is_supported=True,
                company_name=company_name,
                company_name_ok=bool(company_name),
                has_signature_or_seal=signature,
                report_type=report_type,
                notes=[],
            )

            # persist compliance into session store for retrieval/debugging
            sess = SESSION_STORE.get(sid)
            sess.file_compliance[fname] = {
                "is_supported": compliance.is_supported,
                "company_name": compliance.company_name,
                "company_name_ok": compliance.company_name_ok,
                "has_signature_or_seal": compliance.has_signature_or_seal,
                "report_type": compliance.report_type,
                "industry": industry,
                "company_size": company_size,
                "region": region,
                "target": target,
                "signature_note": signature_note,
                "notes": compliance.notes,
            }

            page_summaries: List[PageSummary] = []

            # Ensure the session has a place to store per-file page summaries
            sess.page_summaries.setdefault(parsed.filename, {})

            # 1. First, generate quick page summaries for the UI
            for page in parsed.pages:
                summary = summarize_page(page.text)
                page_summaries.append(PageSummary(page_number=page.page_number, summary=summary))
                sess.page_summaries[parsed.filename][page.page_number] = summary
            
            SESSION_STORE.save(sid)

            # 2. Embedding / indexing: use general page-level indexing via `add_documents`.
            docs: List[Document] = []
            for page in parsed.pages:
                docs.append(
                    Document(
                        page_content=page.text,
                        metadata={
                            "filename": parsed.filename,
                            "page_number": page.page_number,
                            "summary": sess.page_summaries[parsed.filename].get(page.page_number, ""),
                        },
                    )
                )
            add_documents(sess, docs)

            SESSION_STORE.save(sid)
        except Exception:
            import traceback
            traceback.print_exc()
            # reload session to avoid stale data
            sess = SESSION_STORE.get(sid)
            # record a note so clients can see embedding/indexing failed
            sess.file_compliance[fname] = sess.file_compliance.get(fname, {})
            sess.file_compliance[fname]["notes"] = ["processing_failed"]
            SESSION_STORE.save(sid)

    background_tasks.add_task(_process_upload, session_id, str(file.filename or ""), content)

    compliance = ComplianceResult(is_supported=True, notes=["processing"])
    return DocumentIngestion(filename=str(file.filename or ""), page_summaries=[], compliance=compliance)


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
                    tp = int(target_page or 1)
                    image_path = Path("/tmp") / f"{uuid.uuid4()}-page-{tp}.jpg"
                    render_pdf_page_to_jpg(pdf_bytes, tp, image_path)
                    try:
                        vl_answer = call_vl_llm(image_path, question)
                        # Only append if VL actually found something useful and doesn't contradict a strong text answer
                        if vl_answer and "未找到" not in vl_answer and "无法确定" not in vl_answer:
                            answer_text = f"{answer_text}\n\n(视觉核对结果 [第{tp}页]: {vl_answer})"
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


    # Persist answers into session.rag_answers so frontend can display suggestions.
    # Mapping strategy:
    # 1) If client provided explicit `keys` (backwards-compatible), use them.
    # 2) If session.questionnaire exists and lengths match, map by index.
    # 3) Try exact question-text matching.
    # 4) Fallback to generated keys `rag_auto_N`.
    try:
        keys = getattr(payload, "keys", None)
        if isinstance(keys, list) and len(keys) == len(answers):
            for i, k in enumerate(keys):
                key = str(k)
                ans = answers[i]
                session.rag_answers[key] = {
                    "answer": ans.answer,
                    "extracted_answer": ans.extracted_answer,
                    "summary": ans.summary,
                    "sources": [
                        {"filename": src.filename, "page_number": src.page_number, "summary": src.summary}
                        for src in ans.sources
                    ],
                }
        else:
            q_items = session.questionnaire or []
            if q_items and len(q_items) == len(answers):
                for i, item in enumerate(q_items):
                    key = str(item.get("key") or f"q_{i}")
                    ans = answers[i]
                    session.rag_answers[key] = {
                        "answer": ans.answer,
                        "extracted_answer": ans.extracted_answer,
                        "summary": ans.summary,
                        "sources": [
                            {"filename": src.filename, "page_number": src.page_number, "summary": src.summary}
                            for src in ans.sources
                        ],
                    }
            else:
                for ans in answers:
                    mapped = False
                    for item in session.questionnaire or []:
                        q_text = item.get("question")
                        if isinstance(q_text, str) and isinstance(ans.question, str) and ans.question.strip() == q_text.strip():
                            key = str(item.get("key") or q_text)
                            session.rag_answers[key] = {
                                "answer": ans.answer,
                                "extracted_answer": ans.extracted_answer,
                                "summary": ans.summary,
                                "sources": [
                                    {"filename": src.filename, "page_number": src.page_number, "summary": src.summary}
                                    for src in ans.sources
                                ],
                            }
                            mapped = True
                            break
                    if not mapped:
                        gen_key = f"rag_auto_{len(session.rag_answers)}"
                        session.rag_answers[gen_key] = {
                            "question": ans.question,
                            "answer": ans.answer,
                            "extracted_answer": ans.extracted_answer,
                            "summary": ans.summary,
                            "sources": [
                                {"filename": src.filename, "page_number": src.page_number, "summary": src.summary}
                                for src in ans.sources
                            ],
                        }
        SESSION_STORE.save(payload.session_id)
    except Exception:
        # Don't let persistence failures break the response; just continue
        pass

    return QuestionResponse(session_id=payload.session_id, answers=answers)


@app.post("/questionnaire/parse", response_model=QuestionnaireResponse)
async def parse_questionnaire_api(file: UploadFile = File(...)) -> QuestionnaireResponse:
    """
    Standalone API to convert an attached file (PDF, DOCX, TXT, MD) into a structured ESG questionnaire.
    Input: Multipart form-data with 'file' field.
    Output: {"items": [{"key": "...", "type": "...", "question": "...", "options": [...]}, ...]}
    """
    content = await file.read()
    try:
        parsed = parse_document(str(file.filename or ""), content)
        text = "\n".join(page.text for page in parsed.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    system_prompt = (
        "你是一个专业的 ESG 问卷解析助手。你的任务是将非结构化的文档内容转换为结构化的 JSON 格式问卷。"
    )

    user_prompt = (
        "请解析以下文档内容，并将其转换为一个 JSON 列表。每个对象包含以下字段：\n"
        "1. `key`: 一个唯一的、描述性的英文键名（如 `env_policy`, `scope1_emissions`）。\n"
        "2. `type`: 题目类型，必须是以下之一：'单选', '多选', '数值', '百分比', '文本'。\n"
        "   - 如果题目包含选项列表，通常是 '单选' 或 '多选'。\n"
        "   - 如果题目询问数值、总量、占比或包含单位（如 kWh, 吨, %），请设为 '数值' 或 '百分比'。\n"
        "   - 否则设为 '文本'。\n"
        "3. `question`: 题目文本（去掉前面的数字编号）。\n"
        "4. `options`: 如果是单选或多选，列出所有选项的列表；否则为 null。\n\n"
        "待解析内容：\n"
        f"{text}\n\n"
        "输出要求：\n"
        "- 仅输出纯 JSON 列表，不要包含任何 Markdown 代码块标签或解释文字。\n"
        "- 确保 JSON 格式正确。"
    )

    try:
        resp = call_text_llm(user_prompt, system_prompt=system_prompt)
        resp = resp.strip()
        if resp.startswith("```json"): resp = resp[7:]
        if resp.startswith("```"): resp = resp[3:]
        if resp.endswith("```"): resp = resp[:-3]
        resp = resp.strip()
        
        items = json.loads(resp)
        if not isinstance(items, list):
            raise ValueError("LLM did not return a list")
            
        return QuestionnaireResponse(items=items)  # type: ignore[arg-type]
    except Exception:
        # Fallback to regex parser if LLM fails (only works well for MD/TXT)
        items = parse_questionnaire(text)
        return QuestionnaireResponse(items=items)  # type: ignore[arg-type]


@app.post("/session/{session_id}/questionnaire/upload-markdown")
async def upload_questionnaire_markdown(session_id: str, file: UploadFile = File(...)) -> dict:
    """
    Upload a file, parse it, and save it directly to the session.
    """
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    # Reuse the parsing logic
    result = await parse_questionnaire_api(file)
    
    session.questionnaire = [item.dict() for item in result.items]
    SESSION_STORE.save(session_id)
    return {"items": session.questionnaire}


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

@app.get("/session/{session_id}/rag-ready")
def get_rag_ready(session_id: str) -> dict:
    """Return whether the RAG/vectorstore for the given session is ready.

    Response: {"session_id": str, "rag_ready": bool, "has_vectorstore": bool}
    """
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    ready = session.vectorstore is not None
    return {"session_id": session_id, "rag_ready": ready, "has_vectorstore": ready}


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


@app.post("/compliance/detect-signature")
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