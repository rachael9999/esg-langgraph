from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pdfplumber
import fitz
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass
class PageContent:
    page_number: int
    text: str


@dataclass
class ParsedDocument:
    filename: str
    pages: list[PageContent]


COMPANY_PATTERN = re.compile(r"([\w\u4e00-\u9fff]+(?:公司|集团|有限责任公司|股份有限公司))")


SIGNATURE_KEYWORDS = ["签名", "签字", "公司章", "盖章", "印章"]


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def detect_company_name(text: str) -> str | None:
    match = COMPANY_PATTERN.search(text)
    return match.group(1) if match else None


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
