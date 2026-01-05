from __future__ import annotations

import re
from typing import Iterable, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

from .llm import call_text_llm, call_embedding, build_embeddings, QwenConfigError
from .session_store import SessionData


class RagState(TypedDict):
    question: str
    rewritten_question: Optional[str]
    retrieved_docs: Optional[List[Document]]
    answer: Optional[str]
    extracted_answer: Optional[str]


def ensure_vectorstore(session: SessionData) -> FAISS:
    if session.vectorstore is None:
        session.vectorstore = FAISS.from_texts(["ESG Assistant Initialized"], embedding=build_embeddings())
    return session.vectorstore


def add_documents(session: SessionData, docs: List[Document]) -> None:
    vectorstore = ensure_vectorstore(session)
    vectorstore.add_documents(docs)


def rewrite_question(state: RagState) -> RagState:
    question = state["question"].strip()
    prompt = f"请改写问题，使其更适合基于公司文件检索：{question}"
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
            "请根据以下多个来源的检索内容回答问题，给出简洁明确的中文答复。\n"
            "如果不同来源有冲突或补充，请在回答中予以体现。请尽量综合所有来源的信息进行回答。\n"
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
    if not answer or answer == "未找到相关内容。":
        return {**state, "extracted_answer": ""}

    prompt = (
        "请从以下回答中提取出最直接的答案（如数值、日期、具体名称等）。\n"
        "如果是选择题（单选或多选），请严格只返回匹配的可选项文本，多个选项用分号';'分隔。不要包含任何解释、序号或'适用'等字样。\n"
        f"回答：{answer}"
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
    # The second argument to a node function is the config dictionary
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
