from __future__ import annotations

import io
import re
import uuid
import time
import os
import tempfile
import json
import base64
import requests
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pdfplumber
import fitz
from docx import Document

from .llm import call_vl_llm

# Baidu OCR credentials: prefer environment variables for safety
BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")

if not BAIDU_API_KEY or not BAIDU_SECRET_KEY:
    # Try one more time in case load_dotenv was delayed
    from dotenv import load_dotenv
    load_dotenv()
    BAIDU_API_KEY = os.getenv("BAIDU_API_KEY")
    BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY")

_baidu_token_cache: dict = {"token": None, "expires_at": 0}


def _get_baidu_access_token() -> str | None:
    now = time.time()
    token = _baidu_token_cache.get("token")
    if token and now < _baidu_token_cache.get("expires_at", 0) - 10:
        return token

    if not BAIDU_API_KEY or not BAIDU_SECRET_KEY:
        print("ERROR: BAIDU_API_KEY or BAIDU_SECRET_KEY not set.")
        return None

    url = (
        f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={urllib.parse.quote_plus(BAIDU_API_KEY)}&client_secret={urllib.parse.quote_plus(BAIDU_SECRET_KEY)}"
    )
    try:
        resp = requests.post(url, headers={"Content-Type": "application/json"}, timeout=10)
        data = resp.json()
        if "error" in data:
            print(f"Baidu Token Error: {data.get('error_description')}")
            return None
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 2592000))
        if token:
            _baidu_token_cache["token"] = token
            _baidu_token_cache["expires_at"] = now + expires_in
            return token
    except Exception as e:
        print(f"Baidu Access Token Exception: {e}")
        return None
    return None


def _call_baidu_seal_api(image_b64: str) -> dict | None:
    token = _get_baidu_access_token()
    if not token:
        return None
    url = f"https://aip.baidubce.com/rest/2.0/ocr/v1/seal?access_token={token}"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"image": image_b64}
    try:
        resp = requests.post(url, data=data, headers=headers, timeout=15)
        return resp.json()
    except Exception:
        return None


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass
class PageContent:
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    filename: str
    pages: list[PageContent]


COMPANY_PATTERN = re.compile(r"([\w\u4e00-\u9fff]{2,60}?(?:公司|集团|有限责任公司|股份有限公司))")


SIGNATURE_KEYWORDS = ["签名", "签字", "公司章", "盖章", "印章"]


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def detect_company_name(text: str) -> str | None:
    """Improved company name detection with heuristics.

    Strategy:
    1. Look for explicit labels like '公司名称：' or '报告单位：'.
    2. Look for short lines that end with common company suffixes.
    3. Fallback to a looser regex but filter out long or clearly non-name matches.
    """
    if not text:
        return None

    # 1) labeled fields (prefer these)
    label_patterns = [
        r"(?:公司名称|企业名称|报告单位|报告机构)\s*[:：]\s*(.+)",
        r"(?:单位名称)\s*[:：]\s*(.+)",
    ]
    for pat in label_patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            # take up to end of line
            candidate = candidate.splitlines()[0].strip()
            if 2 <= len(candidate) <= 80:
                # avoid generic phrases
                if "本报告" in candidate or "信息和数据" in candidate:
                    continue
                return candidate

    # 2) look for short lines that end with company suffix
    lines = text.splitlines()
    for line in lines:
        ln = line.strip()
        if not ln or len(ln) > 120:
            continue
        m = re.search(r"^(.{1,80}?)(公司|集团|有限责任公司|股份有限公司)$", ln)
        if m:
            candidate = m.group(0).strip()
            if not candidate.startswith("本报告"):
                return candidate

    # 3) fallback: looser regex but filter
    matches = COMPANY_PATTERN.findall(text)
    for match in matches:
        if not match:
            continue
        if len(match) > 80:
            continue
        if match.startswith("本报告") or "信息和数据" in match:
            continue
        return match

    return None


def detect_report_type(text: str) -> str | None:
    """Heuristic-based report type detection.

    Returns a short label like 'ESG报告', '年度报告', '社会责任报告',
    '审计报告', or None if unknown.
    """
    if not text:
        return None

    t = text.lower()

    # ESG / sustainability keywords
    if "esg" in t or "可持续发展" in t or "可持续" in t or "sustainability" in t:
        return "ESG报告"

    # 社会责任 / CSR
    if "社会责任报告" in t or "csr" in t:
        return "社会责任报告"

    # 年度报告 / 年报
    if "年度报告" in t or "年报" in t:
        return "年度报告"

    # 审计 / 财务相关
    if ("审计报告" in t or "审计意见" in t) or ("资产负债表" in t and "利润表" in t):
        return "财务/审计报告"

    # 气候 / 碳排放专题
    if "碳排放" in t or "碳中和" in t or "气候" in t:
        return "气候/碳排放报告"

    # environmental / social keywords without explicit 'report' — assume ESG
    if any(k in t for k in ("环境", "社会", "治理", "排放", "可再生")) and "报告" in t:
        return "ESG报告"

    return None


def detect_signature_or_seal(text: str) -> bool:
    return any(keyword in text for keyword in SIGNATURE_KEYWORDS)


def parse_pdf(filename: str, content: bytes) -> ParsedDocument:
    pages: list[PageContent] = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(PageContent(page_number=index, text=text))
    return ParsedDocument(filename=filename, pages=pages)


def parse_docx(filename: str, content: bytes) -> ParsedDocument:
    doc = Document(io.BytesIO(content))
    full_text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    pages = [PageContent(page_number=1, text=full_text)]
    return ParsedDocument(filename=filename, pages=pages)


def parse_txt(filename: str, content: bytes) -> ParsedDocument:
    text = content.decode("utf-8", errors="ignore")
    return ParsedDocument(filename=filename, pages=[PageContent(page_number=1, text=text)])


def parse_document(filename: str, content: bytes) -> ParsedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(filename, content)
    if suffix == ".docx":
        return parse_docx(filename, content)
    if suffix == ".txt":
        return parse_txt(filename, content)
    raise ValueError(f"Unsupported file type: {suffix}")


def summarize_text(text: str, max_len: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return "无可用文本用于总结。"
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}..."


def iter_pages_for_summary(pages: list[PageContent]) -> Iterator[tuple[int, str]]:
    for page in pages:
        yield page.page_number, summarize_text(page.text)


def render_pdf_page_to_jpg(content: bytes, page_number: int, output_path: Path) -> Path:
    doc = fitz.open(stream=content, filetype="pdf")
    page = doc.load_page(page_number - 1)
    pix = page.get_pixmap(dpi=200)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(str(output_path))
    doc.close()
    return output_path


def get_pdf_page_count(content: bytes) -> int:
    """Quickly get the number of pages in a PDF without full parsing."""
    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            return doc.page_count
    except Exception:
        return 0


def parse_questionnaire(content: str) -> list[dict]:
    items = []
    # 更加宽松的分割逻辑，支持多种换行符和可能的空格
    # 寻找形如 - `key` (type) 的行作为起始
    blocks = re.split(r"\n\s*-\s+`", "\n" + content)
    for block in blocks:
        if not block.strip():
            continue

        lines = block.splitlines()
        if not lines:
            continue
            
        header = lines[0].strip()
        # 匹配 key` (type)
        header_match = re.match(r"([^`]+)`\s*\(([^)]+)\)", header)
        if not header_match:
            continue

        key, q_type = header_match.groups()
        question = ""
        options = []

        current_section = None
        for line in lines[1:]:
            stripped = line.strip()
            if not stripped:
                continue
                
            # 匹配 "- 问题：" 或 "问题："
            if stripped.startswith("- 问题：") or stripped.startswith("问题："):
                question = stripped.replace("- 问题：", "").replace("问题：", "").strip()
                current_section = "question"
            # 匹配 "- 选项：" 或 "选项："
            elif stripped.startswith("- 选项：") or stripped.startswith("选项："):
                # options may be inline or followed by bullets
                current_section = "options"
                inline = stripped.replace("- 选项：", "").replace("选项：", "").strip()
                if inline:
                    # split inline options by common delimiters
                    parts = re.split(r"[;；,，\n]+", inline)
                    for p in parts:
                        p = p.strip()
                        if p:
                            options.append(p)
            # 在选项部分，匹配以 - 或 * 开头的行
            elif (current_section == "options" or (stripped.startswith("- ") or stripped.startswith("* ")) and current_section is None):
                # treat bullet lines as options if we're in options section or no section specified
                if stripped.startswith("- ") or stripped.startswith("* "):
                    opt_text = re.sub(r"^[-*]\s+", "", stripped).strip()
                    # skip bullets that are actually '问题：' or sub-headers
                    if opt_text.startswith("问题：") or opt_text.startswith("选项："):
                        continue
                    if opt_text:
                        options.append(opt_text)
            # 如果遇到新的顶级项（虽然 split 已经处理了，但为了保险）
            elif stripped.startswith("- `"):
                break

        if question:
            # normalize type
            qt = q_type.strip()
            norm_type = qt
            if "多选" in qt or "多项" in qt:
                norm_type = "多选"
            elif "单选" in qt or "单项" in qt:
                norm_type = "单选"
            elif "数值" in qt or "数字" in qt:
                norm_type = "数值"
            elif "百分比" in qt or "%" in qt:
                norm_type = "百分比"
            elif "多行" in qt or "多行文本" in qt:
                norm_type = "文本"
            elif "文本" in qt or "字符串" in qt:
                norm_type = "文本"

            items.append(
                {
                    "key": key.strip(),
                    "type": norm_type,
                    "question": question,
                    "options": options if options else None,
                }
            )
    return items


def detect_signature_in_image(image_bytes: bytes) -> tuple[bool, str | None]:
    """Use Baidu Seal API to detect signature/seal in an image."""
    try:
        img_b64 = base64.b64encode(image_bytes).decode()
        resp = _call_baidu_seal_api(img_b64)
        if not resp:
            return False, "Failed to call detection API"
        
        if isinstance(resp, dict):
            if resp.get("error_code"):
                return False, f"API Error: {resp.get('error_msg')}"
            
            if resp.get("result_num", 0) > 0:
                return True, "检测到印章/签名。"
            
            # Fallback to OCR text
            words_result = resp.get("words_result")
            if words_result:
                for item in words_result:
                    if detect_signature_or_seal(item.get("words", "")):
                        return True, "文字中包含签名/盖章关键词。"
    except Exception as e:
        return False, f"Internal Error: {str(e)}"
    
    return False, "未检测到印章或签名。"


def detect_industry(text: str) -> str | None:
    """Try to detect an industry (prefer ISIC-like labels) from text.

    This is a lightweight heuristic: prefer explicit labels, then keyword mapping.
    """
    if not text:
        return None

    # 1) explicit label
    m = re.search(r"(?:所属行业|行业|行业分类)\s*[:：]\s*(.+)", text)
    if m:
        candidate = m.group(1).splitlines()[0].strip()
        return candidate

    t = text.lower()
    mapping = {
        "金融": "K: 金融与保险业",
        "制造": "C: 制造业",
        "农业": "A: 农、林、牧、渔业",
        "软件": "J: 信息传输、软件和信息技术服务业",
        "能源": "B/D: 采矿/电力热力燃气及水生产供应业",
        "建筑": "F: 建筑业",
        "教育": "P: 教育",
        "医疗": "Q: 卫生和社会工作",
        "零售": "G: 批发和零售业",
        "运输": "H: 交通运输、仓储和邮政业",
        "餐饮": "I: 住宿和餐饮业",
    }
    for kw, label in mapping.items():
        if kw in t:
            return label

    return None


def detect_company_size(text: str) -> str | None:
    """Detect employee-size bucket from text.

    Returns one of: '1-49', '50-249', '250-999', '1000+' or None.
    """
    if not text:
        return None

    # look for explicit numeric mentions
    m = re.search(r"员工(?:约|大约|约有)?\s*([0-9,，]+)\s*(名|人)?", text)
    if m:
        num = m.group(1).replace(",", "").replace("，", "")
        try:
            n = int(num)
            if n < 50:
                return "1-49"
            if n < 250:
                return "50-249"
            if n < 1000:
                return "250-999"
            return "1000+"
        except Exception:
            pass

    # textual hints
    if re.search(r"中小企业|中小型企业|中小型公司", text):
        return "1-249"

    return None


def detect_region(text: str) -> str | None:
    """Naive country/region detection using common country names (Chinese/English).
    Returns the matched country name or None.
    """
    if not text:
        return None

    countries = [
        ("中国", "中国"), ("china", "中国"), ("美国", "美国"), ("united states", "美国"),
        ("英国", "英国"), ("united kingdom", "英国"), ("加拿大", "加拿大"), ("日本", "日本"),
        ("德国", "德国"), ("法国", "法国"),
    ]
    t = text.lower()
    for key, name in countries:
        if key.lower() in t:
            return name

    # fallback: look for '所在地' label
    m = re.search(r"(?:所在地|注册地址|所属地区)\s*[:：]\s*(.+)", text)
    if m:
        return m.group(1).splitlines()[0].strip()

    return None


def detect_target(text: str) -> str:
    """Return target field default. Currently defaults to '客户问卷反馈'."""
    return "客户问卷反馈"


def detect_signature_in_pdf(pdf_bytes: bytes, num_pages: int) -> tuple[bool, str | None]:
    """Use VLM to inspect PDF pages for signatures/seals.

    Returns (found: bool, note: Optional[str]). 
    Optimized: Checks first 3 and last 5 pages first, then the rest.
    """
    if not pdf_bytes or num_pages is None or num_pages == 0:
        return False, None

    # Prioritize finding seals near the front (cover/intro) or back (declaration/assurance)
    # Most ESG reports have seals on the last few pages.
    pages_to_check = []
    if num_pages <= 10:
        pages_to_check = list(range(1, num_pages + 1))
    else:
        # First 3
        pages_to_check.extend([1, 2, 3])
        # Last 5
        last_pages = [num_pages - i for i in range(5)]
        for lp in last_pages:
            if lp > 3:
                pages_to_check.append(lp)
        # Remaining pages
        checked = set(pages_to_check)
        for p in range(1, num_pages + 1):
            if p not in checked:
                pages_to_check.append(p)

    for p in pages_to_check:
        tmp_path = None
        try:
            # create a secure temporary file for the rendered page
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp_path = Path(tf.name)
            tf.close()
            
            image_path = render_pdf_page_to_jpg(pdf_bytes, p, tmp_path)
            img_bytes = Path(image_path).read_bytes()
            img_b64 = base64.b64encode(img_bytes).decode()
            resp = _call_baidu_seal_api(img_b64)
            
            if not resp:
                continue
            
            if isinstance(resp, dict):
                error_code = resp.get("error_code")
                if error_code:
                    continue
                
                # Check for successful detection
                # result = resp.get("result")
                words_result = resp.get("words_result") # fallback for some OCR types
                
                if resp.get("result_num", 0) > 0:
                    return True, f"在第 {p} 页检测到印章/签名。"
                elif words_result and len(words_result) > 0:
                    # Fallback: check if any recognized text contains signature keywords
                    for item in words_result:
                        words = item.get("words", "")
                        if detect_signature_or_seal(words):
                            return True, f"在第 {p} 页检测到印章/签名。"
        except Exception:
            continue
        finally:
            if tmp_path and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except:
                    pass

    return False, None
