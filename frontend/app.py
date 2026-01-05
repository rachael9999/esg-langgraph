import uuid
import html
import time
import re

import requests
import json
import math
import urllib3
from requests.exceptions import ReadTimeout, ConnectionError, RequestException
import streamlit as st

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="ESG LangGraph Assistant", layout="wide")


def safe_post(url: str, **kwargs):
    """Wrapper around requests.post that returns (response, error_message).

    On success returns (response, None). On network errors returns (None, str).
    """
    try:
        resp = requests.post(url, **kwargs)
        return resp, None
    except ReadTimeout:
        return None, "请求超时 (Read timeout). 后端可能未启动或响应较慢。"
    except ConnectionError:
        return None, "无法连接到后端 (Connection error)。请确认后端已运行并监听 http://localhost:8000。"
    except RequestException as e:
        return None, f"网络请求失败: {e}"


def parse_numeric(val) -> float:
    """Try to coerce various string/number inputs into a float for number_input.

    Examples handled: '12,500 吨 CO2 当量' -> 12500.0, '35%' -> 35.0, '3.14' -> 3.14
    """
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return 0.0
    if val is None:
        return 0.0
    s = str(val).strip()
    if not s:
        return 0.0
    # remove percent sign but keep number
    s = s.replace('%', '')
    # remove common unit words and chinese characters
    s = re.sub(r"[\s\u4e00-\u9fff,%]+", "", s)
    # now s should contain digits, maybe with commas or dots
    s = s.replace(',', '')
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return 0.0
    try:
        return float(m.group(0))
    except Exception:
        return 0.0


def build_rag_prompt(item: dict) -> str:
    question = item.get("question", "")
    options = item.get("options") or []
    if not options:
        return question

    formatted = "\n".join(f"{idx + 1}. {opt}" for idx, opt in enumerate(options))
    return f"{question}\n可选项：\n{formatted}"

query_params = st.query_params
persisted_session_id = query_params.get("session_id", [None])[0]

if persisted_session_id:
    st.session_state.session_id = persisted_session_id
elif "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "questionnaire" not in st.session_state:
    st.session_state.questionnaire = []
if "rag_answers" not in st.session_state:
    st.session_state.rag_answers = {}
if "ready_new_session" not in st.session_state:
    st.session_state.ready_new_session = False
if "rag_depth" not in st.session_state:
    st.session_state.rag_depth = 1
if "pending_rag_fills" not in st.session_state:
    st.session_state.pending_rag_fills = {}

# Apply pending fills from RAG at the very beginning, before any widgets are rendered
if st.session_state.pending_rag_fills:
    for key, val in st.session_state.pending_rag_fills.items():
        # Find the item to check its type
        item = next((i for i in st.session_state.questionnaire if i["key"] == key), None)
        if item and "多选" in item["type"]:
            # Ensure value is a list for multiselect
            if isinstance(val, str):
                # Try to match options more robustly
                options = item.get("options") or []
                matched = []
                # Split by common delimiters
                parts = re.split(r"[;；,，\s和&]+", val)
                for part in parts:
                    part = part.strip()
                    if not part: continue
                    # 1. Exact match or partial match of option text
                    for opt in options:
                        if opt in part or part in opt:
                            if opt not in matched:
                                matched.append(opt)
                    # 2. Match by index (e.g., "1" matches the first option)
                    if part.isdigit():
                        idx = int(part) - 1
                        if 0 <= idx < len(options):
                            if options[idx] not in matched:
                                matched.append(options[idx])
                
                # 3. Fallback: check if any option text is in the original string
                for opt in options:
                    if opt in val and opt not in matched:
                        matched.append(opt)
                
                st.session_state[f"ans_{key}"] = matched
            else:
                st.session_state[f"ans_{key}"] = val
        else:
            # For numeric types, ensure the value is a float
            if "数值" in item["type"] or "百分比" in item["type"]:
                st.session_state[f"ans_{key}"] = parse_numeric(val)
            else:
                st.session_state[f"ans_{key}"] = val
    st.session_state.pending_rag_fills = {}

st.sidebar.title("Session 管理")

# Fetch existing sessions
try:
    sessions_resp = requests.get(f"{API_BASE}/sessions", timeout=5)
    if sessions_resp.ok:
        existing_sessions = sessions_resp.json()
    else:
        existing_sessions = []
except Exception:
    existing_sessions = []

if st.session_state.session_id not in existing_sessions:
    existing_sessions.insert(0, st.session_state.session_id)

selected_session = st.sidebar.selectbox(
    "选择或输入 Session ID",
    options=existing_sessions,
    index=existing_sessions.index(st.session_state.session_id) if st.session_state.session_id in existing_sessions else 0
)

if selected_session != st.session_state.session_id:
    st.session_state.session_id = selected_session
    try:
        q_resp = requests.get(f"{API_BASE}/session/{selected_session}/questionnaire", timeout=5)
        if q_resp.ok:
            data = q_resp.json()
            st.session_state.questionnaire = data.get("items", [])
            st.session_state.rag_answers = data.get("rag_answers", {})
        else:
            st.session_state.rag_answers = {}
    except Exception:
        st.session_state.rag_answers = {}
    st.rerun()

if st.sidebar.button("新建 Session"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.questionnaire = []
    st.session_state.rag_answers = {}
    st.rerun()
elif not st.session_state.questionnaire:
    try:
        q_resp = requests.get(f"{API_BASE}/session/{selected_session}/questionnaire", timeout=5)
        if q_resp.ok:
            data = q_resp.json()
            st.session_state.questionnaire = data.get("items", [])
            st.session_state.rag_answers = data.get("rag_answers", {})
        else:
            st.session_state.rag_answers = {}
    except Exception:
        st.session_state.rag_answers = {}

st.sidebar.divider()
st.sidebar.title("问卷配置")
st.sidebar.markdown("**当前 Session ID**")
st.sidebar.code(st.session_state.session_id)
q_file = st.sidebar.file_uploader("上传问卷文件 (PDF/DOCX/TXT/MD)", type=["pdf", "docx", "txt", "md"])
if q_file:
    if st.sidebar.button("解析并生成问卷"):
        with st.spinner("正在解析问卷结构..."):
            response, err = safe_post(
                f"{API_BASE}/session/{st.session_state.session_id}/questionnaire/upload-markdown",
                files={"file": (q_file.name, q_file.getvalue())},
                timeout=120
            )
            if err:
                st.sidebar.error(err)
            elif response.ok:
                items = response.json().get("items", [])
                st.session_state.questionnaire = items
                st.sidebar.success(f"成功解析 {len(items)} 个问题")
                st.rerun()
            else:
                st.sidebar.error(f"解析失败: {response.text}")
else:
    st.sidebar.info("请上传包含问卷内容的文档。")

st.title("ESG 文档问答")

st.sidebar.divider()
st.sidebar.subheader("上传文件")
file = st.sidebar.file_uploader("上传 PDF/DOCX/TXT", type=["pdf", "docx", "txt"])

if file and st.sidebar.button("上传到后端"):
    with st.spinner("上传并解析中..."):
        response, err = safe_post(
            f"{API_BASE}/upload",
            data={"session_id": st.session_state.session_id},
            files={"file": (file.name, file.getvalue())},
            timeout=(10, 600),
        )
    if err:
        st.sidebar.error(err)
    elif response.ok:
        payload = response.json()
        st.sidebar.success("上传成功")
        st.sidebar.json(payload)
    else:
        st.sidebar.error(f"上传失败: {response.text}")

    if response and response.ok:
        status_placeholder = st.sidebar.empty()

        waited = 0
        interval = 2
        timeout = 300
        status_placeholder.info("正在索引中，可能需要一些时间，请稍候...")
        while waited < timeout:
            try:
                r = requests.get(f"{API_BASE}/session/{st.session_state.session_id}/status", timeout=5)
                if r.ok:
                    stat = r.json()
                    if stat.get("has_vectorstore"):
                        status_placeholder.success("索引已完成，可以进行检索。")
                        break
            except Exception:
                pass
            time.sleep(interval)
            waited += interval
        else:
            status_placeholder.warning("索引尚未完成，请稍后重试或查看后端日志。")

st.sidebar.subheader("提问")
questions_text = st.sidebar.text_area("每行一个问题", height=150)
st.sidebar.number_input(
    "检索深度 (1-5)",
    min_value=1,
    max_value=5,
    value=st.session_state.rag_depth,
    step=1,
    key="rag_depth",
)
if st.sidebar.button("提交问题"):
    questions = [line.strip() for line in questions_text.splitlines() if line.strip()]
    if not questions:
        st.sidebar.warning("请输入至少一个问题。")
    else:
        status_ok = False
        wait_seconds = 0
        poll_interval = 2
        max_wait = 300
        status_msg = st.sidebar.empty()
        status_msg.info("正在检查索引状态，请稍候...")
        while wait_seconds < max_wait:
            try:
                r = requests.get(f"{API_BASE}/session/{st.session_state.session_id}/status", timeout=5)
                if r.ok:
                    stat = r.json()
                    if stat.get("has_vectorstore"):
                        status_ok = True
                        status_msg.success("索引已完成，开始检索。")
                        break
            except Exception:
                pass
            time.sleep(poll_interval)
            wait_seconds += poll_interval

        if not status_ok:
            status_msg.warning("索引尚未完成，无法检索。请稍后再试或查看后端日志。")
        else:
            with st.spinner("检索中..."):
                response, err = safe_post(
                    f"{API_BASE}/ask",
                    json={
                        "session_id": st.session_state.session_id,
                        "questions": questions,
                        "depth": int(st.session_state.rag_depth),
                    },
                    timeout=(10, 600),
                )
            if err:
                st.sidebar.error(err)
                st.session_state.ask_answers = []
            elif response.ok:
                payload = response.json()
                st.session_state.ask_answers = payload.get("answers", [])
                ans_map = {a["question"]: a for a in st.session_state.ask_answers}
                for item in st.session_state.questionnaire:
                    if item["question"] in ans_map:
                        ans_data = ans_map[item["question"]]
                        st.session_state.rag_answers[item["key"]] = ans_data
                        # Fill the input field with extracted_answer if available
                        extracted = ans_data.get("extracted_answer")
                        if extracted:
                            st.session_state.pending_rag_fills[item["key"]] = extracted
                st.rerun()
            else:
                st.sidebar.error(f"请求失败: {response.text}")
                st.session_state.ask_answers = []

if "ask_answers" not in st.session_state:
    st.session_state.ask_answers = []

if st.session_state.ask_answers:
    for answer in st.session_state.ask_answers:
        st.sidebar.markdown(f"**问题:** {answer['question']}")
        st.sidebar.markdown(f"**答案:** {answer['answer']}")
        if answer.get("sources"):
            st.sidebar.markdown("**来源:**")
            for source in answer["sources"]:
                tooltip = html.escape(source.get("summary") or "")
                st.sidebar.markdown(
                    f"- <span title=\"{tooltip}\">{source['filename']} (第 {source['page_number']} 页)</span>",
                    unsafe_allow_html=True,
                )
        st.sidebar.divider()

st.subheader("问卷调查")
if st.session_state.questionnaire:
    with st.form("questionnaire_form"):
        for item in st.session_state.questionnaire:
            key = item["key"]
            q_type = item["type"]
            question = item["question"]
            options = item.get("options")

            st.markdown(f"**{question}**")
            
            if key in st.session_state.rag_answers:
                ans_data = st.session_state.rag_answers[key]
                extracted = ans_data.get("extracted_answer")
                if extracted:
                    st.success(f"✅ 提取答案：{extracted}")
                st.info(f"建议答案: {ans_data['answer']}")
                if ans_data.get("sources"):
                    st.markdown("**来源参考**")
                    for source in ans_data["sources"]:
                        tooltip = html.escape(source.get("summary") or "")
                        st.markdown(
                            f"· {source['filename']} (第 {source['page_number']} 页){' — ' + tooltip if tooltip else ''}",
                            unsafe_allow_html=True,
                        )

            # Choose widget based on normalized type
            state_key = f"ans_{key}"
            # Ensure default exists in session_state with appropriate type
            if state_key not in st.session_state:
                if "多选" in q_type:
                    default_val = []
                elif "数值" in q_type or "百分比" in q_type:
                    default_val = 0.0
                elif "单选" in q_type:
                    default_val = options[0] if options else ""
                else:
                    default_val = ""
                st.session_state[state_key] = default_val

            if "多选" in q_type:
                if options:
                    default = st.session_state.get(state_key, [])
                    if not isinstance(default, list):
                        default = [default] if default else []
                    st.session_state[state_key] = default
                    st.multiselect(f"选择 ({key})", options, default=default, key=state_key)
                else:
                    # No options provided; allow comma-separated input then parse later
                    st.text_area(f"多选 - 手动输入选项，逗号分隔 ({key})", key=state_key)
            elif "单选" in q_type:
                if options:
                    default = st.session_state.get(state_key, None)
                    # attempt to set initial index if default exists
                    try:
                        idx = options.index(default) if default in options else 0
                        st.radio(f"选择 ({key})", options, index=idx, key=state_key)
                    except Exception:
                        st.radio(f"选择 ({key})", options, key=state_key)
                else:
                    st.text_input(f"单选 - 手动输入 ({key})", key=state_key)
            elif "百分比" in q_type:
                # render as number input 0-100
                val = parse_numeric(st.session_state.get(state_key, 0.0))
                # Force session state to be float to avoid TypeError with number_input
                st.session_state[state_key] = val
                st.number_input(f"百分比 ({key})", min_value=0.0, max_value=100.0, value=val, step=1.0, format="%.1f", key=state_key)
            elif "数值" in q_type:
                # render as numeric input; allow floats
                val = parse_numeric(st.session_state.get(state_key, 0.0))
                # Force session state to be float to avoid TypeError with number_input
                st.session_state[state_key] = val
                st.number_input(f"数值 ({key})", value=val, key=state_key)
            else:
                # 默认文本，多行
                default = st.session_state.get(state_key, "")
                st.text_area(f"输入内容 ({key})", value=default, key=state_key)

        submit_rag = st.form_submit_button("提交问卷并询问 RAG")

    if submit_rag:
        prompt_map: dict[str, str] = {}
        questions_for_rag: list[str] = []
        for item in st.session_state.questionnaire:
            prompt = build_rag_prompt(item)
            questions_for_rag.append(prompt)
            prompt_map[prompt] = item["key"]

        with st.spinner("正在根据问卷检索 RAG..."):
            response = requests.post(
                f"{API_BASE}/ask",
                json={
                    "session_id": st.session_state.session_id,
                    "questions": questions_for_rag,
                    "depth": int(st.session_state.rag_depth),
                },
                timeout=(10, 600),
            )
        if response.ok:
            payload = response.json()
            for answer in payload.get("answers", []):
                key = prompt_map.get(answer.get("question", ""))
                if key:
                    st.session_state.rag_answers[key] = answer
                    # Fill the input field with extracted_answer if available
                    extracted = answer.get("extracted_answer")
                    if extracted:
                        st.session_state.pending_rag_fills[key] = extracted
            st.rerun()
        else:
            st.error(f"请求失败: {response.text}")
else:
    st.info("暂无解析问卷，请先在侧边栏上传问卷文件并点击 '解析并生成问卷'。")
