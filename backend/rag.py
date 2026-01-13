from __future__ import annotations

import re
from typing import Iterable, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langgraph.graph import StateGraph, END
import faiss

from .llm import call_text_llm, call_embedding, build_embeddings, QwenConfigError, EMBED_DIM
from .session_store import SessionData


class RagState(TypedDict):
    question: str
    rewritten_question: Optional[str]
    retrieved_docs: Optional[List[Document]]
    answer: Optional[str]
    extracted_answer: Optional[str]


def ensure_vectorstore(session: SessionData) -> FAISS:
    if session.vectorstore is None:
        # Create compressed FAISS index using IVF-PQ
        d = EMBED_DIM  # dimension, e.g., 1536
        nlist = 100  # number of clusters for IVF
        m = 16  # number of sub-quantizers for PQ (d must be divisible by m)
        nbits = 8  # bits per sub-vector (2^8 = 256 centroids per sub-vector)
        
        # Use Inner Product (cosine similarity)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        
        # Initialize the index (train on dummy data if needed, but FAISS handles it)
        # For PQ, we need to train, but for simplicity, assume embeddings are added later
        
        session.vectorstore = FAISS(build_embeddings(), index=index, docstore={}, index_to_docstore_id={})
    return session.vectorstore


def add_documents(session: SessionData, docs: List[Document]) -> None:
    vectorstore = ensure_vectorstore(session)
    
    # Get embeddings for the documents
    embeddings = build_embeddings()
    texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)
    
    # Convert to numpy array
    import numpy as np
    vectors_np = np.array(vectors, dtype=np.float32)
    
    # Train the index if not trained
    if not vectorstore.index.is_trained:
        vectorstore.index.train(vectors_np)
    
    # Add vectors to the index
    ids = vectorstore.index.add(vectors_np)
    
    # Update docstore and index_to_docstore_id
    for doc, doc_id in zip(docs, ids):
        doc_id_str = str(doc_id)
        vectorstore.docstore[doc_id_str] = doc
        vectorstore.index_to_docstore_id[doc_id] = doc_id_str
        
        # Save vector_id to doc metadata
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["vector_id"] = doc_id_str


def rewrite_question(state: RagState) -> RagState:
    question = state["question"].strip()
    prompt = f"请改写问题，使其更适合基于公司文件检索：{question}, 对于范围一二三排放优先使用市场基础法（market based）计算，请将此条写入优化后的问题中，如果有绿点抵消，比如碳汇或碳信用额度，也请在问题中体现。"
    try:
        rewritten = call_text_llm(prompt)
    except QwenConfigError:
        rewritten = f"请根据上传的公司文件回答：{question}"
    return {
        "question": question,
        "rewritten_question": rewritten,
        "retrieved_docs": state.get("retrieved_docs"),
        "answer": state.get("answer"),
        "extracted_answer": state.get("extracted_answer"),
    }


def retrieve_docs(state: RagState, vectorstore: FAISS) -> RagState:
    query = state.get("rewritten_question") or state["question"]
    docs = vectorstore.similarity_search(query, k=4)
    return {
        "question": state["question"],
        "rewritten_question": state.get("rewritten_question"),
        "retrieved_docs": docs,
        "answer": state.get("answer"),
        "extracted_answer": state.get("extracted_answer"),
    }


def answer_question(state: RagState) -> RagState:
    retrieved_docs = state.get("retrieved_docs")
    if not retrieved_docs:
        answer = "未找到相关内容。"
    else:
        # Format docs with source info to help LLM distinguish between files and pages
        formatted_docs = []
        for i, doc in enumerate(retrieved_docs):
            meta = doc.metadata or {}
            fname = meta.get("filename", "未知文件")
            pnum = meta.get("page_number", "未知页")
            formatted_docs.append(f"--- 来源 {i+1}: {fname} (第 {pnum} 页) ---\n{doc.page_content}")
        
        combined = "\n\n".join(formatted_docs)
        prompt = (
            "请根据以下多个来源的检索内容回答问题，给出详尽且准确的中文答复。\n"
            "如果不同来源有冲突或补充，请在回答中予以体现。请尽量综合所有来源的信息进行回答。\n"
            "对于简答题，请提供完整的段落描述；对于选择题，请明确指出选中的选项及其理由。\n"
            f"问题：{state['question']}\n"
            f"检索内容：\n{combined}"
        )
        try:
            answer = call_text_llm(prompt)
        except QwenConfigError:
            answer = combined[:500] if combined else "未找到相关内容。"
    return {
        "question": state["question"],
        "rewritten_question": state.get("rewritten_question"),
        "retrieved_docs": retrieved_docs,
        "answer": answer,
        "extracted_answer": state.get("extracted_answer"),
    }


def generate_input(state: RagState) -> RagState:
    answer = state.get("answer")
    question = state.get("question")
    if not answer or answer == "未找到相关内容。":
        return {**state, "extracted_answer": ""}

    prompt = (
        f"原始问题：{question}\n"
        f"详细回答：{answer}\n\n"
        "任务：请从“详细回答”中提取出最直接、准确且完整的答案，用于填入表格。\n"
        "要求：\n"
        "1. 如果原始问题是选择题（包含 1.xxx 2.xxx 等选项），请务必返回选项对应的完整文本内容。如果回答中只提到了序号（如'1'或'A'），请将其转换为对应的选项文本。多个选项用分号';'分隔。\n"
        "2. 如果是简答题，请提取出最详尽的相关段落或具体信息，确保答案完整且具有代表性，不要只返回一个词或短语，除非那是唯一的答案。\n"
        "3. 严禁包含任何解释性文字、序号（除非序号是选项文本的一部分）或引导词（如“答案是：”）。\n"
        "4. 如果无法从回答中提取出有效答案，请返回空字符串。"
    )
    try:
        extracted = call_text_llm(prompt).strip()
    except QwenConfigError:
        extracted = ""
    
    return {
        **state,
        "extracted_answer": extracted,
    }


def build_rag_graph() -> StateGraph:
    # Cache the constructed StateGraph to avoid adding duplicate state keys
    if hasattr(build_rag_graph, "_cached_graph") and build_rag_graph._cached_graph is not None:
        return build_rag_graph._cached_graph

    graph = StateGraph(RagState)
    graph.add_node("rewrite_query", rewrite_question)
    graph.add_node("retrieve_docs", lambda state, config: retrieve_docs(state, config["configurable"]["vectorstore"]))
    graph.add_node("generate_answer", answer_question)
    graph.add_node("generate_input", generate_input)
    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve_docs")
    graph.add_edge("retrieve_docs", "generate_answer")
    graph.add_edge("generate_answer", "generate_input")
    graph.add_edge("generate_input", END)

    build_rag_graph._cached_graph = graph
    return graph


def run_rag(question: str, session: SessionData) -> RagState:
    vectorstore = ensure_vectorstore(session)
    graph = build_rag_graph().compile()
    initial: RagState = {
        "question": question,
        "rewritten_question": None,
        "retrieved_docs": None,
        "answer": None,
        "extracted_answer": None,
    }
    result: RagState = graph.invoke(initial, {"configurable": {"vectorstore": vectorstore}})
    return result


def extract_modules(question: str, answer: str) -> List[str]:
    prompt = (
        "请根据问题和答案，概括需要继续深入检索的模块或主题，"
        "只返回用逗号分隔的关键词，不要解释。\n"
        f"问题：{question}\n"
        f"答案：{answer}"
    )
    try:
        response = call_text_llm(prompt)
    except QwenConfigError:
        return []
    parts = re.split(r"[,\n，;；、]+", response)
    modules = [item.strip() for item in parts if item.strip()]
    return modules[:5]


def dedupe_docs(docs: Iterable[Document]) -> List[Document]:
    seen: set[tuple[str, int, str]] = set()
    deduped: List[Document] = []
    for doc in docs:
        metadata = doc.metadata or {}
        key = (
            str(metadata.get("filename", "")),
            int(metadata.get("page_number", 0)),
            doc.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(doc)
    return deduped


def run_rag_with_depth(question: str, session: SessionData, depth: int = 1) -> RagState:
    vectorstore = ensure_vectorstore(session)
    graph = build_rag_graph().compile()
    initial: RagState = {
        "question": question,
        "rewritten_question": None,
        "retrieved_docs": None,
        "answer": None,
        "extracted_answer": None,
    }
    state: RagState = graph.invoke(initial, {"configurable": {"vectorstore": vectorstore}})
    merged_docs = state.get("retrieved_docs") or []

    for _ in range(1, max(depth, 1)):
        modules = extract_modules(question, state.get("answer") or "")
        if not modules:
            break
        extra_docs: List[Document] = []
        for module in modules:
            module_question = f"{question}（关注模块：{module}）"
            module_state: RagState = graph.invoke(
                {
                    "question": module_question,
                    "rewritten_question": None,
                    "retrieved_docs": None,
                    "answer": None,
                    "extracted_answer": None,
                },
                {"configurable": {"vectorstore": vectorstore}},
            )
            if module_state.get("retrieved_docs"):
                extra_docs.extend(module_state["retrieved_docs"])
        merged_docs = dedupe_docs([*merged_docs, *extra_docs])
        state = answer_question(
            {
                "question": question,
                "rewritten_question": state.get("rewritten_question"),
                "retrieved_docs": merged_docs,
                "answer": None,
                "extracted_answer": None,
            }
        )
        # After re-answering with merged docs, we should also re-extract the input
        state = generate_input(state)

    return {
        "question": question,
        "rewritten_question": state.get("rewritten_question"),
        "retrieved_docs": merged_docs,
        "answer": state.get("answer"),
        "extracted_answer": state.get("extracted_answer"),
    }


def is_numeric_question(question: str) -> bool:
    keywords = ["金额", "数值", "多少", "总额", "比例", "比率", "百分比"]
    if any(keyword in question for keyword in keywords):
        return True
    return bool(re.search(r"\d", question))


def delete_documents_by_filename(session: SessionData, filename: str) -> None:
    """Delete documents from vectorstore by filename."""
    from .database import SessionLocal, DBDocument
    from .session_store import SESSION_STORE
    import json
    
    db = SessionLocal()
    try:
        doc_record = db.query(DBDocument).filter(
            DBDocument.session_id == session.session_id,
            DBDocument.filename == filename
        ).first()
        
        if doc_record and doc_record.vector_ids:
            vector_ids = json.loads(doc_record.vector_ids)
            vectorstore = ensure_vectorstore(session)
            vectorstore.delete(vector_ids)
            SESSION_STORE.save_vectorstore(session)
        
        # Remove from DB
        if doc_record:
            db.delete(doc_record)
            db.commit()
    finally:
        db.close()
    """Rebuild the vectorstore for the session, excluding documents with the given filename."""
    from .database import SessionLocal, DBDocument
    from .utils import parse_document, summarize_page
    from .session_store import SESSION_STORE
    
    db = SessionLocal()
    try:
        docs = db.query(DBDocument).filter(DBDocument.session_id == session.session_id).all()
        
        # Create new vectorstore
        embeddings = build_embeddings()
        new_vectorstore = FAISS.from_texts(["ESG Assistant Initialized"], embedding=embeddings)
        
        for doc_record in docs:
            if doc_record.filename == exclude_filename:
                continue  # Skip the excluded file
            

            pass

        if docs and any(d.filename != exclude_filename for d in docs):

            print(f"Warning: Cannot rebuild vectorstore without original file contents. Removing {exclude_filename} from records only.")
            session.vectorstore = None  # Force reload or something
        else:
            session.vectorstore = new_vectorstore
        
        SESSION_STORE.save_vectorstore(session)
        
    finally:
        db.close()
