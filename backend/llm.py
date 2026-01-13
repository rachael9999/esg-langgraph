from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
import urllib3
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

# Load environment variables from .env file
load_dotenv()

# Suppress InsecureRequestWarning if SSL verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

QWEN_API_KEY = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
if not QWEN_API_KEY:
    print("WARNING: QWEN_API_KEY or DASHSCOPE_API_KEY is not set. Please set it in your .env file.")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
TEXT_MODEL = os.getenv("QWEN_TEXT_MODEL", "qwen-flash")
VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen3-vl-flash")
MOCK_LLM = os.getenv("MOCK_LLM", "0").lower() in ("1", "true", "yes")
SSL_VERIFY = os.getenv("SSL_VERIFY", "1").lower() in ("1", "true", "yes")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

# Global session for connection pooling
_session = requests.Session()


class QwenConfigError(RuntimeError):
    pass


def _require_key() -> str:
    key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise QwenConfigError("QWEN_API_KEY 或 DASHSCOPE_API_KEY 未配置。请在 .env 文件中设置。")
    return key


def _post_chat(payload: Dict[str, Any]) -> str:
    api_key = _require_key()
    url = f"{QWEN_BASE_URL}/chat/completions"
    
    for attempt in range(3):
        try:
            response = _session.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=600,
                verify=SSL_VERIFY,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content")
            # content may be a string or a structured list (Dashscope returns structured content for multimodal)
            if isinstance(content, list):
                texts = [p.get("text") for p in content if isinstance(p, dict) and p.get("text")]
                return "\n".join(texts) if texts else str(content)
            return str(content)
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as exc:
            if attempt == 2:
                raise QwenConfigError(f"LLM request failed after retries: {exc}") from exc
            time.sleep(1)
        except requests.exceptions.HTTPError as exc:
            # Retry on 404 as requested, or 429/5xx
            if attempt < 2 and (exc.response.status_code == 404 or exc.response.status_code >= 500 or exc.response.status_code == 429):
                time.sleep(1)
                continue
            raise QwenConfigError(f"LLM request failed: {exc}") from exc
        except requests.RequestException as exc:
            # Network/SSL or other requests-level errors — wrap and propagate as config/runtime error
            raise QwenConfigError(f"LLM request failed: {exc}") from exc
        except (KeyError, ValueError) as exc:
            # Unexpected response format
            raise QwenConfigError(f"Unexpected LLM response format: {exc}") from exc
    
    return "" # Should not reach here


def call_embedding(text: str) -> list[float]:
    if MOCK_LLM:
        # return a deterministic pseudo-embedding for local testing
        # use a simple hash-based generator to produce EMBED_DIM floats in [-0.01,0.01]
        import hashlib

        h = hashlib.sha256(text.encode("utf-8")).digest()
        vals = []
        i = 0
        while len(vals) < EMBED_DIM:
            # expand hash deterministically
            block = hashlib.sha256(h + i.to_bytes(4, "little")).digest()
            for b in block:
                if len(vals) >= EMBED_DIM:
                    break
                vals.append((b / 255.0 - 0.5) * 0.02)
            i += 1
        return vals
    api_key = _require_key()
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    payload = {
        "model": "text-embedding-v2",
        "input": {"texts": [text]},
        "parameters": {"text_type": "query"},
    }
    
    for attempt in range(3):
        try:
            response = _session.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=60,
                verify=SSL_VERIFY,
            )
            response.raise_for_status()
            data = response.json()
            return data["output"]["embeddings"][0]["embedding"]
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as exc:
            if attempt == 2:
                raise QwenConfigError(f"Embedding request failed after retries: {exc}") from exc
            time.sleep(1)
        except requests.exceptions.HTTPError as exc:
            if attempt < 2 and (exc.response.status_code == 404 or exc.response.status_code >= 500 or exc.response.status_code == 429):
                time.sleep(1)
                continue
            raise QwenConfigError(f"Embedding request failed: {exc}") from exc
        except Exception as exc:
            raise QwenConfigError(f"Embedding request failed: {exc}") from exc
    
    return [] # Should not reach here


def call_text_llm(prompt: str, system_prompt: str | None = None) -> str:
    # Dashscope compatible message format: messages[].content should be a structured list
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
    return _post_chat({"model": TEXT_MODEL, "messages": messages, "temperature": 0.2})

def call_vl_llm(image_b64: str, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    return _post_chat({"model": VL_MODEL, "messages": messages, "temperature": 0.1})

def summarize_page(text: str) -> str:
    prompt = (
        "请将以下页面内容总结为不超过120字的中文摘要，突出关键数据与结论：\n"
        f"{text}"
    )
    try:
        return call_text_llm(prompt)
    except QwenConfigError:
        return text.strip()[:240] or "无可用文本用于总结。"


class DashScopeEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [call_embedding(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return call_embedding(text)


def build_embeddings() -> Embeddings:
    return DashScopeEmbeddings()
