import uuid
import html
import time
import re
import requests
import json
import urllib3
from requests.exceptions import ReadTimeout, ConnectionError, RequestException
import streamlit as st

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_BASE = "http://localhost:8000"

def init_session():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "questionnaire" not in st.session_state:
        st.session_state.questionnaire = []
    if "rag_answers" not in st.session_state:
        st.session_state.rag_answers = {}
    if "rag_depth" not in st.session_state:
        st.session_state.rag_depth = 2

def safe_post(url: str, **kwargs):
    try:
        resp = requests.post(url, **kwargs)
        return resp, None
    except Exception as e:
        return None, str(e)

def build_rag_prompt(item: dict) -> str:
    question = item.get("question", "")
    options = item.get("options") or []
    if not options: return question
    formatted = "\n".join(f"{idx + 1}. {opt}" for idx, opt in enumerate(options))
    return f"{question}\n可选项：\n{formatted}"

def main():
    st.set_page_config(page_title="ESG system", layout="wide", page_icon="")
    init_session()

    # --- Sidebar ---
    st.sidebar.title(" 控制面板")
    
    with st.sidebar.expander(" 会话管理", expanded=True):
        try:
            sessions = requests.get(f"{API_BASE}/sessions").json()
        except: sessions = []
        
        if st.session_state.session_id not in sessions:
            sessions.insert(0, st.session_state.session_id)
            
        sel = st.selectbox("当前会话 ID", sessions, index=sessions.index(st.session_state.session_id))
        if sel != st.session_state.session_id:
            st.session_state.session_id = sel
            st.rerun()
            
        if st.button(" 开启新会话"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.questionnaire = []
            st.session_state.rag_answers = {}
            st.rerun()

    with st.sidebar.expander(" 数据导入", expanded=False):
        st.markdown("**1. 问卷结构**")
        q_file = st.file_uploader("上传问卷 (Markdown/Docx/PDF)", type=["md", "docx", "pdf", "txt"], key="q_up")
        if q_file and st.button("解析问卷内容", use_container_width=True):
            with st.spinner("结构提取中..."):
                safe_post(f"{API_BASE}/session/{st.session_state.session_id}/questionnaire/upload-markdown", files={"file": (q_file.name, q_file.getvalue())})
                st.rerun()

        st.divider()
        st.markdown("**2. 参考文件 (索引)**")
        e_file = st.file_uploader("上传参考文件 (PDF/TXT)", type=["pdf", "txt"], key="e_up")
        if e_file and st.button("上传并开始索引", use_container_width=True):
            with st.spinner("上传并开始索引..."):
                safe_post(f"{API_BASE}/upload", data={"session_id": st.session_state.session_id}, files={"file": (e_file.name, e_file.getvalue())})
                st.info("任务已提交，正在后台索引。")

    with st.sidebar.expander(" 检索设置", expanded=False):
        st.session_state.rag_depth = st.slider("检索 Chunk 数量", 1, 10, st.session_state.rag_depth)
        st.caption("增加该值可提高回答全面性，但会增加 Token 消耗。")

    # --- Main Screen ---
    st.title(" ESG 专家系统")    

    tab_q, tab_c = st.tabs([" 自动化问卷", " 合规校验"])

    with tab_q:
        st.subheader("问卷智能填充")
        # Sync with backend
        try:
            q_data = requests.get(f"{API_BASE}/session/{st.session_state.session_id}/questionnaire").json()
            st.session_state.questionnaire = q_data.get("items", [])
            st.session_state.rag_answers = q_data.get("rag_answers", {})
        except: pass

        if not st.session_state.questionnaire:
            st.warning("暂无问卷结构，请在左侧侧边栏上传。")
        else:
            # Use a session-unique form key to avoid duplicate widget IDs across sessions
            with st.form(key=f"q_filling_form_{st.session_state.session_id}"):
                for it in st.session_state.questionnaire:
                    key = it["key"]
                    question = it["question"]
                    qtype = it["type"]
                    st.markdown(f"**[{key}] {question}** ({qtype})")
                    
                    if key in st.session_state.rag_answers:
                        ans_obj = st.session_state.rag_answers[key]
                        st.info(f" AI 建议: {ans_obj.get('answer', '暂无建议')}")
                        if ans_obj.get('extracted_answer'):
                            st.code(f"提取结果: {ans_obj['extracted_answer']}", language="text")

                    inp_key = f"inp_{key}"
                    # Prefill the input with AI suggestion when available
                    ai_sugg = None
                    if key in st.session_state.rag_answers:
                        ai_sugg = st.session_state.rag_answers[key].get("answer")
                    if inp_key not in st.session_state and ai_sugg is not None:
                        st.session_state[inp_key] = ai_sugg

                    opts = it.get("options") or []
                    qtype = it.get("type", "")

                    if opts and qtype == "单选":
                        try:
                            default_idx = opts.index(st.session_state.get(inp_key, opts[0]))
                        except Exception:
                            default_idx = 0
                        st.radio("", opts, index=default_idx, key=inp_key, label_visibility="collapsed")
                    elif opts and qtype == "多选":
                        default_sel = st.session_state.get(inp_key) if isinstance(st.session_state.get(inp_key), list) else []
                        st.multiselect("", opts, default=default_sel, key=inp_key, label_visibility="collapsed")
                    elif opts:
                        try:
                            default_idx = opts.index(st.session_state.get(inp_key, opts[0]))
                        except Exception:
                            default_idx = 0
                        st.selectbox("", opts, index=default_idx, key=inp_key, label_visibility="collapsed")
                    else:
                        if qtype in ("数值", "百分比"):
                            # try parse numeric suggestion
                            parsed_val = None
                            v = st.session_state.get(inp_key)
                            try:
                                parsed_val = float(v) if v not in (None, "") else 0.0
                            except Exception:
                                parsed_val = 0.0
                            st.number_input("", value=parsed_val, key=inp_key, label_visibility="collapsed")
                        else:
                            st.text_area("点击确认并微调结果", placeholder="AI 建议将显示在此处上方...", key=inp_key)
                    st.divider()
                
                if st.form_submit_button(" 启动全量 AI 检索填充", use_container_width=True):
                    with st.spinner("AI 正在深度检索索引..."):
                        qs = [build_rag_prompt(it) for it in st.session_state.questionnaire]
                        safe_post(
                            f"{API_BASE}/ask",
                            json={
                                "session_id": st.session_state.session_id,
                                "questions": qs,
                                "depth": st.session_state.rag_depth,
                            },
                        )
                    st.rerun()

    with tab_c:
        st.subheader("自动化合规审计")
        try:
            c_resp = requests.get(f"{API_BASE}/session/{st.session_state.session_id}/compliance")
            if c_resp.ok:
                c_info = c_resp.json()
                for fn, details in c_info.items():
                    with st.container(border=True):
                        st.markdown(f"####  {fn}")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            company_name = details.get("company_name", "未知")
                            report_type = details.get("report_type", "未知")
                            industry = details.get("industry", "未知")
                            company_size = details.get("company_size", "未知")
                            st.write(f" 企业名称: **{company_name}**")
                            st.write(f" 报告类型: **{report_type}**")
                            st.write(f" 所属行业: **{industry}**")
                            st.write(f" 企业规模: **{company_size}**")
                        with col_b:
                            seal = details.get("has_signature_or_seal", False)
                            region = details.get("region", "未知")
                            target = details.get("target", "客户问卷反馈")
                            st.write(f" 签章识别: {' 已识别' if seal else ' 未发现'}")
                            st.write(f" 所属地区: **{region}**")
                            st.write(f" 目标: **{target}**")
                            sig_note = details.get("signature_note")
                            if sig_note:
                                st.caption(f"签章识别说明: {sig_note}")
                        
                        if details.get("notes"):
                            st.info(" 系统备注: " + " | ".join(details["notes"]))
        except:
            st.error("暂无合规分析结果。")

if __name__ == "__main__":
    main()
