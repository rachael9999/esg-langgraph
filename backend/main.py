from __future__ import annotations

import re
import uuid
import json
import requests
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, BackgroundTasks
from fastapi import Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document

from .llm import summarize_page, call_text_llm
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
    wait: bool = Form(False)
) -> DocumentIngestion:
    """Accept file upload. 
    Handles company info extraction and seal detection.
    """
    content = await file.read()
    filename = str(file.filename or "unknown")

    if not is_supported(filename):
        raise HTTPException(status_code=400, detail="文件格弝丝支挝。")

    session = SESSION_STORE.get(session_id)
    if filename not in session.files:
        session.files.append(filename)
    session.file_contents[filename] = content
    SESSION_STORE.save(session_id)

    def _process_upload_task(sid: str, fname: str, data: bytes) -> None:
        _process_upload(sid, fname, data)

    if wait:
        _process_upload_task(session_id, filename, content)
        # Refresh session to get updated info
        session = SESSION_STORE.get(session_id)
        comp_dict = session.file_compliance.get(filename, {})
        compliance = ComplianceResult(**comp_dict)
        
        page_summaries = []
        file_summaries = session.page_summaries.get(filename, {})
        for pnum, summ in sorted(file_summaries.items()):
            page_summaries.append(PageSummary(page_number=pnum, summary=summ))

        return DocumentIngestion(
            filename=filename,
            page_summaries=page_summaries,
            compliance=compliance,
        )
    else:
        background_tasks.add_task(_process_upload_task, session_id, filename, content)
        return DocumentIngestion(
            filename=filename,
            page_summaries=[],
            compliance=ComplianceResult(is_supported=True, notes=["解析任务已提交。"]),
        )

@app.post("/session/{session_id}/upload-and-index")
async def upload_and_index(session_id: str, file: UploadFile = File(...)) -> dict:
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
        
    # Save file content
    if filename not in session.files:
        session.files.append(filename)
    session.file_contents[filename] = content

    try:
        # Parse and summarize
        parsed = parse_document(filename, content)
        session.page_summaries.setdefault(filename, {})
        
        docs = []
        for page in parsed.pages:
            # Generate a quick summary for metadata
            summary = summarize_page(page.text)
            session.page_summaries[filename][page.page_number] = summary
            
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
            
        # Add to FAISS
        add_documents(session, docs)
        SESSION_STORE.save(session_id)
        
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


def _process_upload(sid: str, fname: str, data: bytes) -> None:
    sess = SESSION_STORE.get(sid)
    try:
        parsed = parse_document(fname, data)
        full_text = "\n".join(page.text for page in parsed.pages)

        # 0. Extract Company Name as keyword (Using AI/LLM)
        try:
            prompt = f"请从以下文本内容中提取公司的正式全称。仅输出公司名称，不要包含任何其他解释文字。如果无法确定，请返回 null。\n\n文本内容：\n{full_text[:4000]}"
            ai_name = call_text_llm(prompt).strip()
            if "null" in ai_name.lower() or not ai_name:
                extracted_name = detect_company_name(full_text)
            else:
                extracted_name = ai_name.replace('"', '').replace("'", "").replace("`", "").strip()
        except Exception:
            extracted_name = detect_company_name(full_text)

        # 1. Call external company info API (192.168.1.100)
        company_name = extracted_name
        industry = None
        company_size = None
        region = None
        external_notes = []

        if extracted_name:
            try:
                resp = requests.post(
                    "http://192.168.1.100:8000/pack/companyinfo",
                    json={"keyword": extracted_name},
                    headers={"appkey": "zhongTanDify"},
                    timeout=10
                )
                if resp.status_code == 200:
                    outer = resp.json()
                    if outer.get("success") and outer.get("data"):
                        info = outer["data"]
                        company_name = info.get("name") or company_name
                        industry_all = info.get("industryAll") or {}
                        industry = industry_all.get("categoryBig")
                        tags = info.get("tags") or ""
                        if "小微企业" in tags: company_size = "小微企业"
                        elif "中型企业" in tags: company_size = "中型企业"
                        region = info.get("city") or info.get("district")
                        external_notes.append("External company info fetched successfully.")
            except Exception as e:
                external_notes.append(f"External API call failed: {str(e)}")

        if not industry: industry = detect_industry(full_text)
        if not company_size: company_size = detect_company_size(full_text)
        if not region: region = detect_region(full_text)
        target = "客户问卷反馈"
        # 2. Seal Detection: call internal logic directly
        signature = False
        signature_note = None
        try:
            suffix = Path(fname).suffix.lower()
            if suffix == ".pdf":
                num_pages = get_pdf_page_count(data)
                signature, signature_note = detect_signature_in_pdf(data, num_pages)
            elif suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
                signature, signature_note = detect_signature_in_image(data)
            else:
                signature = detect_signature_or_seal(full_text)
                signature_note = "Text-based detection" if signature else None
        except Exception as e:
            signature_note = f"Seal detection error: {str(e)}"

        report_type = detect_report_type(full_text)

        compliance = ComplianceResult(
            is_supported=True,
            company_name=company_name,
            company_name_ok=bool(company_name),
            has_signature_or_seal=signature,
            report_type=report_type,
            industry=industry,
            company_size=company_size,
            region=region,
            target=target,
            signature_note=signature_note,
            notes=external_notes,
        )

        # persist compliance into session store
        sess = SESSION_STORE.get(sid)
        sess.file_compliance[fname] = compliance.dict()

        # Generate quick page summaries
        sess.page_summaries.setdefault(fname, {})
        for page in parsed.pages:
            summary = summarize_page(page.text)
            sess.page_summaries[fname][page.page_number] = summary
        
        SESSION_STORE.save(sid)
        # FAISS indexing is now moved to index_document endpoint
    except Exception:
        import traceback
        traceback.print_exc()
        # reload session to avoid stale data
        sess = SESSION_STORE.get(sid)
        # record a note so clients can see processing failed
        sess.file_compliance[fname] = sess.file_compliance.get(fname, {})
        sess.file_compliance[fname]["notes"] = ["processing_failed"]
        SESSION_STORE.save(sid)


def best_source(docs: List[Document]) -> AnswerSource | None:
    if not docs:
        return None
    metadata = docs[0].metadata
    return AnswerSource(
        filename=metadata.get("filename", "unknown"),
        page_number=int(metadata.get("page_number", 1)),
    )


# @app.post("/ask", response_model=QuestionResponse)
# async def ask_questions(payload: QuestionRequest) -> QuestionResponse:
#     session = SESSION_STORE.get(payload.session_id)
#     answers: List[AnswerItem] = []

#     for question in payload.questions:
#         rag_state = run_rag_with_depth(question, session, depth=payload.depth)
#         sources: List[AnswerSource] = []
#         summary: str | None = None
#         retrieved_docs = rag_state.get("retrieved_docs")
#         if retrieved_docs:
#             for doc in retrieved_docs:
#                 metadata = doc.metadata or {}
#                 filename = metadata.get("filename", "unknown")
#                 try:
#                     page_num = int(metadata.get("page_number", 1))
#                 except Exception:
#                     page_num = 1
#                 # prefer summary from document metadata, fallback to session page_summaries
#                 summary_text = metadata.get("summary") or session.page_summaries.get(filename, {}).get(page_num)
#                 sources.append(
#                     AnswerSource(
#                         filename=filename,
#                         page_number=page_num,
#                         summary=summary_text,
#                     )
#                 )
#             # Combine summaries from all sources instead of just the first one
#             all_summaries = []
#             for src in sources:
#                 if src.summary:
#                     all_summaries.append(f"[{src.filename} 第{src.page_number}页]: {src.summary}")
#             summary = "\n".join(all_summaries) if all_summaries else None

#         answer_text = rag_state.get("answer") or "未找到相关内容。"
#         if is_numeric_question(question) and retrieved_docs:
#             # Try to find the most relevant page mentioned in the text answer
#             target_page = None
#             target_filename = None
            
#             # Heuristic: look for "第 X 页" or "第X页" in the answer text
#             page_matches = re.findall(r"第\s*(\d+)\s*页", answer_text)
#             if page_matches:
#                 target_page = int(page_matches[0])
#                 # Find which filename this page belongs to from retrieved_docs
#                 for doc in retrieved_docs:
#                     if doc.metadata.get("page_number") == target_page:
#                         target_filename = doc.metadata.get("filename")
#                         break
            
#             # Fallback to best_source if no page mentioned or not found in docs
#             if not target_page or not target_filename:
#                 primary_source = best_source(retrieved_docs)
#                 if primary_source:
#                     target_page = primary_source.page_number
#                     target_filename = primary_source.filename

#             if target_filename and target_filename.lower().endswith(".pdf"):
#                 pdf_bytes = session.file_contents.get(target_filename)
#                 if pdf_bytes:
#                     tp = int(target_page or 1)
#                     image_path = Path("/tmp") / f"{uuid.uuid4()}-page-{tp}.jpg"
#                     render_pdf_page_to_jpg(pdf_bytes, tp, image_path)
#                     try:
#                         vl_answer = call_vl_llm(image_path, question)
#                         # Only append if VL actually found something useful and doesn't contradict a strong text answer
#                         if vl_answer and "未找到" not in vl_answer and "无法确定" not in vl_answer:
#                             answer_text = f"{answer_text}\n\n(视觉核对结果 [第{tp}页]: {vl_answer})"
#                     except QwenConfigError:
#                         pass

#         answers.append(
#             AnswerItem(
#                 question=question,
#                 answer=answer_text,
#                 extracted_answer=rag_state.get("extracted_answer"),
#                 sources=sources,
#                 summary=summary,
#             )
#         )

#     stored_answers = []
#     for ans in answers:
#         stored_answers.append(
#             {
#                 "question": ans.question,
#                 "answer": ans.answer,
#                 "extracted_answer": ans.extracted_answer,
#                 "summary": ans.summary,
#                 "sources": [
#                     {
#                         "filename": src.filename,
#                         "page_number": src.page_number,
#                         "summary": src.summary,
#                     }
#                     for src in ans.sources
#                 ],
#             }
#         )
#     SESSION_STORE.save_rag_answers(payload.session_id, stored_answers)


#     # Persist answers into session.rag_answers so frontend can display suggestions.
#     # Mapping strategy:
#     # 1) If client provided explicit `keys` (backwards-compatible), use them.
#     # 2) If session.questionnaire exists and lengths match, map by index.
#     # 3) Try exact question-text matching.
#     # 4) Fallback to generated keys `rag_auto_N`.
#     try:
#         keys = getattr(payload, "keys", None)
#         if isinstance(keys, list) and len(keys) == len(answers):
#             for i, k in enumerate(keys):
#                 key = str(k)
#                 ans = answers[i]
#                 session.rag_answers[key] = {
#                     "answer": ans.answer,
#                     "extracted_answer": ans.extracted_answer,
#                     "summary": ans.summary,
#                     "sources": [
#                         {"filename": src.filename, "page_number": src.page_number, "summary": src.summary}
#                         for src in ans.sources
#                     ],
#                 }
#         else:
#             q_items = session.questionnaire or []
#             if q_items and len(q_items) == len(answers):
#                 for i, item in enumerate(q_items):
#                     key = str(item.get("key") or f"q_{i}")
#                     ans = answers[i]
#                     session.rag_answers[key] = {
#                         "answer": ans.answer,
#                         "extracted_answer": ans.extracted_answer,
#                         "summary": ans.summary,
#                         "sources": [
#                             {"filename": src.filename, "page_number": src.page_number, "summary": src.summary}
#                             for src in ans.sources
#                         ],
#                     }
#             else:
#                 for ans in answers:
#                     mapped = False
#                     for item in session.questionnaire or []:
#                         q_text = item.get("question")
#                         if isinstance(q_text, str) and isinstance(ans.question, str) and ans.question.strip() == q_text.strip():
#                             key = str(item.get("key") or q_text)
#                             session.rag_answers[key] = {
#                                 "answer": ans.answer,
#                                 "extracted_answer": ans.extracted_answer,
#                                 "summary": ans.summary,
#                                 "sources": [
#                                     {"filename": src.filename, "page_number": src.page_number, "summary": src.summary}
#                                     for src in ans.sources
#                                 ],
#                             }
#                             mapped = True
#                             break
#                     if not mapped:
#                         gen_key = f"rag_auto_{len(session.rag_answers)}"
#                         session.rag_answers[gen_key] = {
#                             "question": ans.question,
#                             "answer": ans.answer,
#                             "extracted_answer": ans.extracted_answer,
#                             "summary": ans.summary,
#                             "sources": [
#                                 {"filename": src.filename, "page_number": src.page_number, "summary": src.summary}
#                                 for src in ans.sources
#                             ],
#                         }
#         SESSION_STORE.save(payload.session_id)
#     except Exception:
#         # Don't let persistence failures break the response; just continue
#         pass

#     return QuestionResponse(session_id=payload.session_id, answers=answers)


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
        "请解析以下文档内容，并将其转换为一个 JSON 列表。每个对象应包含以下字段：\n"
        "1. `key`: 一个唯一的、具有描述性的英文键名（如 `env_policy`, `carbon_emissions`）。\n"
        "2. `type`: 题目类型，必须是以下之一：'单选', '多选', '数值', '百分比', '文本'。\n"
        "   - 如果题目包含明确的选项列表，设为 '单选' 或 '多选'。\n"
        "   - 如果题目要求填写具体数值、总量或百分比，设为 '数值' 或 '百分比'。\n"
        "   - 其他情况设为 '文本'。\n"
        "3. `question`: 题目文本内容（请移除原文档中的数字编号）。\n"
        "4. `options`: 对于单选或多选题目，请列出所有可选项的列表；否则设为 null。\n\n"
        "待解析内容：\n"
        f"{text}\n\n"
        "输出要求：\n"
        "- 必须仅输出纯 JSON 格式的列表，不要包含 Markdown 代码块标签（如 ```json）或任何解释性文字。\n"
        "- 确保 JSON 语法完全正确且可被解析。"
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


@app.post("/session/{session_id}/questionnaire/extract")
async def extract_questionnaire(session_id: str, file: UploadFile = File(...)) -> dict:
    """
    Upload a questionnaire file, parse it into structured questions, and save to the session.
    """
    session = SESSION_STORE.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    # Reuse the parsing logic from parse_questionnaire_api
    result = await parse_questionnaire_api(file)
    
    session.questionnaire = [item.dict() for item in result.items]
    SESSION_STORE.save(session_id)
    return {"items": session.questionnaire, "status": "extracted"}


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
    return {
        "session_id": session_id,
        "rag_ready": ready,
        "has_vectorstore": ready,
        "faiss_last_updated": session.faiss_last_updated,
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