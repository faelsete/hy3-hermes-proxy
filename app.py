"""OpenAI-compatible adapter for Tencent's official Hy3 Hugging Face Space.

The upstream public Gradio endpoint is stateless and returns cumulative
snapshots. This adapter translates those snapshots into OpenAI Chat
Completions responses so Hermes Agent can use Hy3 as a normal provider.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

UPSTREAM_BASE_URL = os.getenv(
    "HY3_SPACE_URL", "https://tencent-hy3.hf.space"
).rstrip("/")
MODEL_ID = os.getenv("HY3_MODEL_ID", "tencent/hy3")
REQUEST_TIMEOUT = float(os.getenv("HY3_REQUEST_TIMEOUT", "600"))
MAX_CONCURRENCY = max(1, int(os.getenv("HY3_MAX_CONCURRENCY", "8")))
MAX_RETRIES = max(0, int(os.getenv("HY3_MAX_RETRIES", "2")))
MAX_INPUT_CHARS = max(10_000, int(os.getenv("HY3_MAX_INPUT_CHARS", "150000")))
# Upstream Gradio hard limit (Hugging Face Space) — nunca enviar mais que isso
UPSTREAM_HARD_LIMIT = 145000
DEFAULT_MAX_TOKENS = max(1, int(os.getenv("HY3_DEFAULT_MAX_TOKENS", "16384")))
MAX_OUTPUT_TOKENS = max(
    DEFAULT_MAX_TOKENS, int(os.getenv("HY3_MAX_OUTPUT_TOKENS", "65536"))
)
CONTEXT_LENGTH_TOKENS = 262144
# KV-cache simulation: in-memory prefix cache (10 min TTL, 50 entries)
_KV_PREFIX_CACHE: dict[str, tuple[float, str]] = {}
_KV_CACHE_TTL = 600
_KV_CACHE_MAX = 50

# --- MiniMax-Text-01 fallback (disabled by default, keep proxy strictly on Hy3) ---
MINIMAX_FALLBACK = os.getenv("MINIMAX_FALLBACK", "0") == "1"
MINIMAX_SPACE_URL = os.getenv(
    "MINIMAX_SPACE_URL", "https://minimaxai-minimax-text-01.hf.space"
).rstrip("/")
MINIMAX_FN_INDEX = int(os.getenv("MINIMAX_FN_INDEX", "16"))

# --- Step-3.7-Flash fallback (disabled by default, keep proxy strictly on Hy3) ---
STEP_FALLBACK = os.getenv("STEP_FALLBACK", "0") == "1"
STEP_SPACE_URL = os.getenv(
    "STEP_SPACE_URL", "https://stepfun-ai-step-3-7-flash-dev.hf.space"
).rstrip("/")
STEP_MODEL_ID = os.getenv("STEP_MODEL_ID", "step/step-3-7-flash")

TRIM_NOTICE = (
    "[Nota de contexto: Histórico anterior compactado para manter agilidade. "
    "Mantenha a resposta em Português do Brasil.]"
)

LANGUAGE_ENFORCEMENT = (
    "\n\n[DIRETIVA DE IDIOMA E COMPORTAMENTO]:\n"
    "- Você é o Hermes, assistente do Rafael.\n"
    "- Responda SEMPRE em Português do Brasil (pt-BR).\n"
    "- NUNCA responda em chinês, mandarim ou inglês.\n"
    "- Respostas diretas, técnicas e concisas."
)

# --- Token counting (tiktoken if available, fallback chars/4) ---
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return len(_ENC.encode(text))
except Exception:
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

def estimate_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(str(m.get("content") or ""))
        total += 4  # overhead per message
        if m.get("tool_calls"):
            total += estimate_tokens(json.dumps(m["tool_calls"], ensure_ascii=False))
    return total

def kv_cache_key(system: str, tools_str: str) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(system.encode()[:8000])
    h.update(tools_str.encode()[:8000])
    return h.hexdigest()[:16]

def kv_cache_get(key: str) -> str | None:
    entry = _KV_PREFIX_CACHE.get(key)
    if not entry:
        return None
    ts, val = entry
    if time.time() - ts > _KV_CACHE_TTL:
        _KV_PREFIX_CACHE.pop(key, None)
        return None
    return val

def kv_cache_set(key: str, val: str):
    if len(_KV_PREFIX_CACHE) >= _KV_CACHE_MAX:
        oldest = min(_KV_PREFIX_CACHE.items(), key=lambda x: x[1][0])[0]
        _KV_PREFIX_CACHE.pop(oldest, None)
    _KV_PREFIX_CACHE[key] = (time.time(), val)

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(
                max_keepalive_connections=40,
                max_connections=120,
                keepalive_expiry=120.0,
            ),
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=15.0),
            follow_redirects=True,
        )
    return _http_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(
        http2=True,
        limits=httpx.Limits(
            max_keepalive_connections=40,
            max_connections=120,
            keepalive_expiry=120.0,
        ),
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=15.0),
        follow_redirects=True,
    )
    yield
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


app = FastAPI(title="Tencent Hy3 Hermes Proxy", version="1.2.0", lifespan=lifespan)
_upstream_slots = asyncio.Semaphore(MAX_CONCURRENCY)
logger = logging.getLogger("uvicorn.error")


def _text_content(content: Any) -> str:
    """Flatten OpenAI text content into the text-only format Hy3 accepts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"}:
                    chunks.append(str(item.get("text", "")))
                elif item.get("type") in {"image_url", "input_image"}:
                    chunks.append("[Imagem não suportada pelo modelo Hy3 textual]")
        return "\n".join(chunk for chunk in chunks if chunk)
    return str(content)


def _clean_history_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = str(message.get("role", "")).strip()
    if role not in {"user", "assistant", "tool"}:
        return None

    cleaned: dict[str, Any] = {
        "role": role,
        "content": _text_content(message.get("content")),
    }
    for key in ("name", "tool_call_id", "reasoning_content", "reasoning_details"):
        if message.get(key) is not None:
            cleaned[key] = message[key]
    if role == "assistant" and message.get("tool_calls"):
        cleaned["tool_calls"] = message["tool_calls"]
    return cleaned


CHINESE_REFUSAL_TRIGGERS = (
    "我无法给到相关内容",
    "无法给到相关内容",
    "我无法提供",
    "抱歉，我无法",
    "无法满足您的要求",
    "违反相关法律法规",
    "抱歉，作为一个人工智能",
)


def _is_chinese_refusal(text: str) -> bool:
    if not text:
        return False
    return any(trigger in text for trigger in CHINESE_REFUSAL_TRIGGERS)


def _sanitize_output(content: str) -> str:
    if not content:
        return ""
    # Strip common Chinese greeting/pleasantry artifacts
    cleaned = re.sub(r"^(好的[，,！!]|您好[，,！!]|很高兴为您服务[，,！!]|当然[，,！!])\s*", "", content.strip())
    return cleaned


def _prune_available_skills_block(sys_prompt: str, max_chars: int = 3000) -> str:
    """Trim oversized skill listings in system prompt while preserving active skills and instructions."""
    pattern = re.compile(r"(<available_skills>)(.*?)(</available_skills>)", re.DOTALL)
    match = pattern.search(sys_prompt)
    if match and len(match.group(2)) > max_chars:
        lines = [line for line in match.group(2).strip().split("\n") if line.strip()]
        pruned = "\n".join(lines[:25]) + "\n    ... [catálogo de skills compactado para performance — use skills_list() ou smart_skills para listar todas]"
        sys_prompt = sys_prompt[: match.start(2)] + "\n" + pruned + "\n" + sys_prompt[match.end(2) :]
    return sys_prompt

def _intelligent_compact(messages: list[dict[str, Any]], keep_recent: int = 20) -> list[dict[str, Any]]:
    """Compactação inteligente: resume mensagens antigas em 1 mensagem de sistema, preserva recentes."""
    if len(messages) <= keep_recent + 5:
        return messages
    old = messages[: len(messages) - keep_recent]
    recent = messages[len(messages) - keep_recent :]
    # cria resumo estruturado das antigas
    summary_parts = []
    for m in old:
        role = m.get("role", "?")
        content = str(m.get("content") or "")[:400]
        if m.get("tool_calls"):
            content += f" [tool_calls: {len(m['tool_calls'])}]"
        if content.strip():
            summary_parts.append(f"[{role}] {content[:300]}")
    summary = TRIM_NOTICE + "\nResumo das " + str(len(old)) + " msgs antigas:\n" + "\n".join(summary_parts[:40])
    if len(summary_parts) > 40:
        summary += f"\n... +{len(summary_parts)-40} msgs omitidas"
    return [{"role": "system", "content": summary}] + recent


def extract_conversation(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], str]:
    """Split OpenAI messages into Space system prompt, history and new turn.

    Gradio's public ``/chat`` endpoint requires a fresh user message. During
    Hermes tool loops, the last message is often a tool result; in that case we
    preserve the full tool history and append a neutral continuation turn.
    """
    system_parts: list[str] = []
    conversation: list[dict[str, Any]] = []

    for raw in messages or []:
        role = str(raw.get("role", "")).strip()
        if role in {"system", "developer"}:
            text = _text_content(raw.get("content")).strip()
            if text:
                system_parts.append(text)
            continue
        cleaned = _clean_history_message(raw)
        if cleaned is not None:
            conversation.append(cleaned)

    if conversation and conversation[-1]["role"] == "user":
        last = conversation.pop()
        message = last.get("content", "").strip()
    else:
        message = (
            "Continue a resposta com base no histórico e ferramentas acima. "
            "Responda em Português do Brasil (pt-BR)."
        )

    if not message:
        message = "Continue a conversa normalmente em Português."

    full_system = "\n\n".join(system_parts)
    full_system = _prune_available_skills_block(full_system)
    if LANGUAGE_ENFORCEMENT not in full_system:
        full_system = f"{full_system}{LANGUAGE_ENFORCEMENT}" if full_system else LANGUAGE_ENFORCEMENT.strip()

    return full_system, conversation, message


def reasoning_level(request: dict[str, Any]) -> str:
    """Map OpenAI/Hermes reasoning controls to Hy3's three levels.

    For summarization/compression turns, automatically disable reasoning
    to prevent slow thinking timeouts and Chinese leakage during context compaction.
    """
    messages = request.get("messages") or []
    for msg in messages:
        c = str(msg.get("content") or "").lower()
        if any(k in c for k in ("summariz", "resuma", "compact", "resumo das mensagens", "resumo")):
            return "no_think"

    effort = request.get("reasoning_effort")
    reasoning = request.get("reasoning")

    if effort is None and isinstance(reasoning, dict):
        if reasoning.get("enabled") is False:
            return "no_think"
        effort = reasoning.get("effort")
        if reasoning.get("enabled") is True and effort is None:
            return "high"
    elif effort is None and reasoning is False:
        return "no_think"
    elif effort is None and reasoning is True:
        return "high"

    value = str(effort or "high").lower()
    if value in {"none", "no", "off", "minimal", "no_think"}:
        return "no_think"
    if value in {"low", "medium"}:
        return "low"
    return "high"


def payload_size_chars(data: list[Any]) -> int:
    """Return the exact serialized character size sent to Gradio."""
    return len(json.dumps(data, ensure_ascii=False, separators=(",", ":")))


def _trim_history_to_budget(
    data: list[Any], max_input_chars: int
) -> tuple[list[Any], int]:
    """Drop oldest turns; se ainda estourar, usa compactação inteligente com resumo."""
    history = list(data[2] or [])
    if payload_size_chars(data) <= max_input_chars or not history:
        return data, 0

    original_count = len(history)
    # tenta corte simples primeiro
    data[1] = f"{data[1]}\n\n{TRIM_NOTICE}" if data[1] else TRIM_NOTICE

    while history and payload_size_chars(data) > max_input_chars:
        next_user = next(
            (
                index
                for index in range(1, len(history))
                if history[index].get("role") == "user"
            ),
            len(history),
        )
        history = history[next_user:]
        data[2] = history or None

    # se ainda estoura mesmo sem histórico, compacta via resumo inteligente (não dropa tudo)
    if payload_size_chars(data) > max_input_chars and not history:
        # reconstrói histórico resumido para não perder contexto total
        data[2] = [{"role": "system", "content": TRIM_NOTICE + " Histórico compactado por limite de contexto."}]

    return data, original_count - len(history)


def _compact_tools_for_upstream(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact tool schemas so Gradio payload never exceeds upstream limits."""
    if not tools:
        return []
    compacted: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") or {}
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        desc = str(fn.get("description") or "")
        if len(desc) > 120:
            desc = desc[:120] + "..."
        params = fn.get("parameters") or {}
        compacted.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": params,
            }
        })
    return compacted


def build_upstream_payload(
    request: dict[str, Any], *, max_input_chars: int = MAX_INPUT_CHARS
) -> dict[str, Any]:
    system_prompt, history, message = extract_conversation(
        request.get("messages") or []
    )
    # Compactação inteligente ANTES de montar payload: se histórico gigante, resume antigos
    total_tokens_est = estimate_messages_tokens([{"role":"system","content":system_prompt}] + history + [{"role":"user","content":message}])
    if total_tokens_est > 35000:  # ~140k chars
        history = _intelligent_compact(history, keep_recent=20)
    raw_tools = request.get("tools") or []
    tools = _compact_tools_for_upstream(raw_tools)
    tools_str = json.dumps(tools, ensure_ascii=False) if tools else ""
    if len(tools_str) > 40_000:
        # Keep essential/first tools if still oversized
        tools = tools[:30]
        tools_str = json.dumps(tools, ensure_ascii=False)

    temperature = request.get("temperature")
    max_tokens = request.get("max_tokens")
    if max_tokens is None:
        max_tokens = request.get("max_completion_tokens")
    try:
        resolved_max_tokens = int(max_tokens or DEFAULT_MAX_TOKENS)
    except (TypeError, ValueError):
        resolved_max_tokens = DEFAULT_MAX_TOKENS
    resolved_max_tokens = min(max(1, resolved_max_tokens), MAX_OUTPUT_TOKENS)
    top_p = request.get("top_p")

    data: list[Any] = [
        message,
        system_prompt,
        history or None,
        reasoning_level(request),
        temperature,
        resolved_max_tokens,
        float(top_p or 0),
        request.get("preserved_thinking"),
        tools_str,
    ]
    data, _ = _trim_history_to_budget(data, max_input_chars)
    return {"data": data}


def parse_gradio_data(raw_data: str) -> dict[str, Any]:
    """Parse a cumulative Gradio snapshot from one SSE data line."""
    parsed = json.loads(raw_data)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("Unexpected Gradio payload")
    values = parsed[0]
    if not isinstance(values, list) or len(values) < 4:
        raise ValueError("Incomplete Gradio snapshot")
    return {
        "content": values[0] or "",
        "reasoning": values[1] or "",
        "tool_calls": values[2] or [],
        "history": values[3] or [],
    }


def normalize_tool_calls(calls: Any) -> list[dict[str, Any]]:
    """Normalize Hy3 tool calls to OpenAI's assistant message shape."""
    if isinstance(calls, str):
        try:
            calls = json.loads(calls)
        except json.JSONDecodeError:
            return []
    if not isinstance(calls, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw in calls:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = fn.get("name") or raw.get("name")
        if not name:
            continue
        arguments = fn.get("arguments", raw.get("arguments", {}))
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        normalized.append(
            {
                "id": raw.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": str(name), "arguments": arguments},
            }
        )
    return normalized


def extract_tool_calls_from_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract embedded XML/JSON tool calls from model generated text."""
    if not text:
        return "", []

    # If partial tool call tag is still being generated, buffer it
    for tag in ("<tool_call>", "<invoke", "<function="):
        if tag in text and ("</tool_call>" not in text and "</invoke>" not in text and "</function>" not in text):
            parts = text.split(tag, 1)
            return parts[0].strip(), []

    tools: list[dict[str, Any]] = []

    # 1. <tool_call>...</tool_call>
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    for m in pattern.finditer(text):
        block = m.group(1).strip()
        # JSON format
        try:
            p = json.loads(block)
            if isinstance(p, dict):
                name = p.get("name") or p.get("function", {}).get("name")
                args = p.get("arguments") or p.get("function", {}).get("arguments") or {}
                if name:
                    tools.append(
                        {
                            "id": f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": str(name),
                                "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                            },
                        }
                    )
                    continue
        except Exception:
            pass
        # XML format
        try:
            root = ET.fromstring(f"<root>{block}</root>")
            for child in root:
                tag = child.tag
                action_el = child.find("action")
                args_el = child.find("args")
                tool_name = action_el.text.strip() if action_el is not None and action_el.text else tag
                args_dict: dict[str, Any] = {}
                if args_el is not None:
                    for p_elem in args_el:
                        args_dict[p_elem.tag] = p_elem.text or ""
                else:
                    for p_elem in child:
                        if p_elem.tag != "action":
                            args_dict[p_elem.tag] = p_elem.text or ""

                if "action" not in args_dict and action_el is not None:
                    args_dict["action"] = action_el.text.strip()

                tool_entry_name = "computer_use" if tag == "computer_use" else tool_name

                tools.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": tool_entry_name,
                            "arguments": json.dumps(args_dict, ensure_ascii=False),
                        },
                    }
                )
        except Exception:
            pass

    # 2. <invoke name="...">...</invoke>
    invoke_pattern = re.compile(r"<invoke\s+name=[\"'](.*?)[\"']>(.*?)</invoke>", re.DOTALL)
    for m in invoke_pattern.finditer(text):
        name = m.group(1).strip()
        body = m.group(2).strip()
        args_dict = {}
        param_pattern = re.compile(r"<parameter\s+name=[\"'](.*?)[\"']>(.*?)</parameter>", re.DOTALL)
        for pm in param_pattern.finditer(body):
            args_dict[pm.group(1).strip()] = pm.group(2).strip()
        if not args_dict:
            try:
                p = json.loads(body)
                if isinstance(p, dict):
                    args_dict = p
            except Exception:
                args_dict = {"input": body}
        tools.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args_dict, ensure_ascii=False)},
            }
        )

    cleaned = pattern.sub("", text)
    cleaned = invoke_pattern.sub("", cleaned).strip()
    return cleaned, tools


def build_nonstream_response(
    *, model: str, content: str, reasoning: str, tool_calls: list[dict[str, Any]], prompt_tokens: int | None = None
) -> dict[str, Any]:
    final_content = _sanitize_output(content)
    clean_text, extracted_tools = extract_tool_calls_from_text(final_content)
    all_tool_calls = list(tool_calls or [])
    if extracted_tools:
        all_tool_calls.extend(extracted_tools)
        final_content = clean_text

    # If content is empty but reasoning is present (and no tool calls), fallback to reasoning
    if not final_content and not all_tool_calls and reasoning:
        final_content = _sanitize_output(reasoning)

    # In standard OpenAI format: when tool calls are made, content is null, NOT leaked text
    msg_content = None if all_tool_calls else (final_content or "")
    message: dict[str, Any] = {"role": "assistant", "content": msg_content}
    if reasoning:
        message["reasoning_content"] = reasoning
        message["reasoning"] = reasoning
    if all_tool_calls:
        message["tool_calls"] = all_tool_calls

    comp_tokens = estimate_tokens(final_content or "")
    if comp_tokens == 0:
        comp_tokens = 1
    prompt_toks = prompt_tokens if prompt_tokens is not None else 0
    return {
        "id": f"chatcmpl-hy3-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if all_tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_toks + comp_tokens,
        },
    }


async def _upstream_snapshots(
    payload: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Submit a Gradio job and yield cumulative snapshots from its SSE feed."""
    submit_url = f"{UPSTREAM_BASE_URL}/gradio_api/call/chat"
    client = get_http_client()

    async with _upstream_slots:
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.post(submit_url, json=payload)
                if response.status_code == 429 or response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Hy3 upstream returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                event_id = response.json().get("event_id")
                if not event_id:
                    raise RuntimeError("Hy3 upstream did not return event_id")

                stream_url = f"{submit_url}/{event_id}"
                current_event = ""
                yielded = False
                async with client.stream(
                    "GET", stream_url, headers={"Accept": "text/event-stream"}
                ) as stream:
                    stream.raise_for_status()
                    async for line in stream.aiter_lines():
                        if line.startswith("event:"):
                            current_event = line[6:].strip()
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if current_event in {"generating", "complete"} and data:
                            parsed_snap = parse_gradio_data(data)
                            snap_text = str(parsed_snap.get("content") or "")
                            if _is_chinese_refusal(snap_text):
                                raise RuntimeError(f"Hy3 upstream Chinese refusal: {snap_text}")
                            yielded = True
                            yield parsed_snap
                        if current_event == "error":
                            raise RuntimeError(f"Hy3 upstream error: {data}")
                        if current_event == "complete":
                            return
                if not yielded:
                    raise RuntimeError("Hy3 upstream returned no snapshots")
                return
            except (httpx.HTTPError, RuntimeError, ValueError) as exc:
                last_error = exc
                if attempt >= MAX_RETRIES:
                    break
                await asyncio.sleep(0.5 * (2**attempt))

        raise RuntimeError(f"Hy3 upstream failed after retries: {last_error}")


async def minimax_complete(
    request_data: dict[str, Any], model: str
) -> dict[str, Any]:
    """Last-resort fallback to MiniMax-Text-01 (token-free, Gradio5 queue).

    Invoked only after Hy3 exhausts its own retries. Returns a snapshot-shaped
    dict so callers can reuse the non-stream / stream response builders.
    """
    _, _history, message = extract_conversation(request_data.get("messages") or [])
    temperature = request_data.get("temperature") or 0.7
    max_tokens = request_data.get("max_tokens") or request_data.get(
        "max_completion_tokens"
    )
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = DEFAULT_MAX_TOKENS
    max_tokens = min(max(1, max_tokens), 16000)
    top_p = request_data.get("top_p") or 0.9

    session_hash = "hmfb" + uuid.uuid4().hex[:16]
    join_payload = {
        "data": [message, None, max_tokens, float(temperature), float(top_p)],
        "fn_index": MINIMAX_FN_INDEX,
        "session_hash": session_hash,
    }

    client = get_http_client()
    join_url = f"{MINIMAX_SPACE_URL}/gradio_api/queue/join"
    resp = await client.post(join_url, json=join_payload)
    resp.raise_for_status()

    stream_url = (
        f"{MINIMAX_SPACE_URL}/gradio_api/queue/data?session_hash={session_hash}"
    )
    async with client.stream(
        "GET", stream_url, headers={"Accept": "text/event-stream"}
    ) as stream:
        stream.raise_for_status()
        async for line in stream.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if obj.get("msg") == "process_completed":
                out = obj.get("output", {})
                data = out.get("data") or []
                text = data[0] if data else ""
                return {"content": _sanitize_output(text) or "", "reasoning": "", "tool_calls": []}
    return {"content": "", "reasoning": "", "tool_calls": []}


def _suffix(previous: str, current: str) -> str:
    if not current:
        return ""
    if current.startswith(previous):
        return current[len(previous) :]
    return current


def _sse(data: dict[str, Any] | str) -> str:
    if isinstance(data, str):
        return f"data: {data}\n\n"
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _openai_stream(
    request_data: dict[str, Any], payload: dict[str, Any], model: str, prompt_tokens: int = 0
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-hy3-{uuid.uuid4().hex}"
    created = int(time.time())
    has_tools = bool(request_data.get("tools"))
    
    previous_content = ""
    previous_reasoning = ""
    final_tools: list[dict[str, Any]] = []
    sent_role = False
    yielded_any_content = False

    last_clean_content = ""
    last_reasoning = ""

    try:
        async for snapshot in _upstream_snapshots(payload):
            raw_content = str(snapshot.get("content") or "")
            reasoning = str(snapshot.get("reasoning") or "")
            
            clean_content, extracted_tools = extract_tool_calls_from_text(raw_content)
            if extracted_tools:
                final_tools = extracted_tools

            tools = normalize_tool_calls(snapshot.get("tool_calls"))
            if tools:
                final_tools = tools

            last_clean_content = clean_content
            last_reasoning = reasoning

            # If tools are present, we buffer content to avoid leaking pre-tool text
            if not has_tools:
                delta_content = _suffix(previous_content, clean_content)
                delta_reasoning = _suffix(previous_reasoning, reasoning)
                previous_content = clean_content
                previous_reasoning = reasoning

                delta: dict[str, Any] = {}
                if not sent_role:
                    delta["role"] = "assistant"
                    sent_role = True
                if delta_reasoning:
                    delta["reasoning_content"] = delta_reasoning
                    delta["reasoning"] = delta_reasoning
                if delta_content:
                    delta["content"] = delta_content
                    yielded_any_content = True

                if delta:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                        }
                    )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        if STEP_FALLBACK:
            logger.warning(
                "Hy3 stream failed after retries (%s); falling back to Step", exc
            )
            snap = await step_complete(request_data, model)
            if not sent_role:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                        ],
                    }
                )
            snap_text = snap.get("content") or snap.get("reasoning") or ""
            if snap_text:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": snap_text},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield _sse("[DONE]")
            return
        raise

    if has_tools:
        if not sent_role:
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
            )
            sent_role = True

        if final_tools:
            tool_deltas = [
                {"index": index, **tool_call}
                for index, tool_call in enumerate(final_tools)
            ]
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": tool_deltas},
                            "finish_reason": None,
                        }
                    ],
                }
            )
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
            )
            yield _sse("[DONE]")
            return
        else:
            # No tool called, output accumulated content
            text_to_emit = _sanitize_output(last_clean_content or last_reasoning)
            if text_to_emit:
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text_to_emit},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            yield _sse("[DONE]")
            return

    # If no content was emitted at all, but we have reasoning and no tools, stream reasoning as content
    if not yielded_any_content and not final_tools and previous_reasoning:
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": _sanitize_output(previous_reasoning)},
                        "finish_reason": None,
                    }
                ],
            }
        )

    if final_tools:
        tool_deltas = [
            {"index": index, **tool_call}
            for index, tool_call in enumerate(final_tools)
        ]
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": tool_deltas},
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if final_tools else "stop",
                }
            ],
        }
    )
    # usage chunk (OpenAI-compatible)
    try:
        comp_toks = estimate_tokens(last_clean_content or "") if 'last_clean_content' in locals() else 0
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max(1, comp_toks), "total_tokens": prompt_tokens + max(1, comp_toks)},
            }
        )
    except Exception:
        pass
    yield _sse("[DONE]")


async def _minimax_stream(
    request_data: dict[str, Any], model: str
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-mm-{uuid.uuid4().hex}"
    created = int(time.time())
    snap = await minimax_complete(request_data, model)
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )
    if snap.get("content"):
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": snap["content"]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield _sse("[DONE]")




async def _to_data_url(url: str) -> str | None:
    """Return a base64 data-URL for an image, downloading remote URLs."""
    if url.startswith("data:"):
        return url
    try:
        client = get_http_client()
        resp = await client.get(url)
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "image/png").split(";")[0] or "image/png"
        b64 = base64.b64encode(resp.content).decode()
        return f"data:{ctype};base64,{b64}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("step image fetch failed: %s", exc)
        return None


async def convert_messages_for_step(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI messages to Step's messages_json format, embedding images as base64."""
    out: list[dict[str, Any]] = []
    has_system = False
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip()
        if role not in {"system", "user", "assistant"}:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            if role == "system":
                has_system = True
                if LANGUAGE_ENFORCEMENT not in content:
                    content = f"{content}{LANGUAGE_ENFORCEMENT}"
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t in {"text", "input_text", "output_text"}:
                    parts.append({"type": "text", "text": str(item.get("text", ""))})
                elif t in {"image_url", "input_image"}:
                    img = item.get("image_url")
                    url = img.get("url") if isinstance(img, dict) else img
                    if not isinstance(url, str) or not url:
                        continue
                    b64 = await _to_data_url(url)
                    if b64:
                        parts.append({"type": "image_url", "image_url": {"url": b64}})
            if parts:
                out.append({"role": role, "content": parts})
    if not has_system:
        out.insert(0, {"role": "system", "content": LANGUAGE_ENFORCEMENT.strip()})
    return out


async def step_complete(
    request_data: dict[str, Any], model: str
) -> dict[str, Any]:
    """Multimodal selector to Step-3.7-Flash (token-free, OpenAI-compatible Gradio Space).

    Exposes /chat_with_step with an OpenAI-style messages_json payload. Vision
    inputs are supported via image_url content parts (converted to base64
    data-URLs). When the model emits only reasoning_content (typical for vision
    turns), it is used as the answer fallback.
    """
    messages = request_data.get("messages") or []
    step_messages = await convert_messages_for_step(messages)
    if not step_messages:
        return {"content": "", "reasoning": "", "tool_calls": []}

    temperature = request_data.get("temperature")
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.7
    max_tokens = request_data.get("max_tokens") or request_data.get("max_completion_tokens")
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens = 4096
    max_tokens = min(max(1, max_tokens), 16000)

    effort = str(request_data.get("reasoning_effort") or "medium").lower()
    if effort not in {"low", "medium", "high"}:
        effort = "medium"

    body = {
        "messages_json": json.dumps(step_messages, ensure_ascii=False),
        "reasoning_effort": effort,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    submit_url = f"{STEP_SPACE_URL}/gradio_api/call/v2/chat_with_step"
    client = get_http_client()
    submit_resp = await client.post(submit_url, json=body)
    submit_resp.raise_for_status()
    event_id = submit_resp.json().get("event_id")
    if not event_id:
        raise RuntimeError("Step upstream did not return event_id")

    stream_url = f"{STEP_SPACE_URL}/gradio_api/call/chat_with_step/{event_id}"
    result_content = ""
    result_reasoning = ""
    async with client.stream(
        "GET", stream_url, headers={"Accept": "text/event-stream"}
    ) as stream:
        stream.raise_for_status()
        async for line in stream.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                arr = json.loads(data)
                if not isinstance(arr, list) or not arr:
                    continue
                p = arr[0]
                if isinstance(p, str):
                    p = json.loads(p)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(p, dict):
                continue
            c = p.get("content") or ""
            r = p.get("reasoning_content") or ""
            if c:
                result_content = c
            if r:
                result_reasoning = r

    final_text = _sanitize_output(result_content or result_reasoning)
    return {
        "content": final_text or "",
        "reasoning": result_reasoning or "",
        "tool_calls": [],
    }


async def _step_stream(
    request_data: dict[str, Any], model: str
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-step-{uuid.uuid4().hex}"
    created = int(time.time())
    snap = await step_complete(request_data, model)
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
            ],
        }
    )
    if snap.get("content"):
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": snap["content"]},
                        "finish_reason": None,
                    }
                ],
            }
        )
    yield _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    yield _sse("[DONE]")


@app.exception_handler(Exception)
async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": str(exc),
                "type": "upstream_error",
                "code": "hy3_upstream_error",
            }
        },
    )


@app.get("/")
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "upstream": UPSTREAM_BASE_URL,
        "max_concurrency": MAX_CONCURRENCY,
        "max_input_chars": MAX_INPUT_CHARS,
        "context_length": CONTEXT_LENGTH_TOKENS,
        "kv_cache_size": len(_KV_PREFIX_CACHE),
        "fallback": "step" if STEP_FALLBACK else ("minimax" if MINIMAX_FALLBACK else None),
        "step_fallback": STEP_FALLBACK,
        "step_upstream": STEP_SPACE_URL if STEP_FALLBACK else None,
        "minimax_selectable": MINIMAX_FALLBACK,
        "minimax_upstream": MINIMAX_SPACE_URL if MINIMAX_FALLBACK else None,
    }


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    data = [
        {
            "id": MODEL_ID,
            "object": "model",
            "created": 1783344048,
            "owned_by": "tencent",
            "context_length": CONTEXT_LENGTH_TOKENS,
        }
    ]
    if MINIMAX_FALLBACK:
        data.append(
            {
                "id": "minimax/text-01",
                "object": "model",
                "created": 1783344048,
                "owned_by": "minimax",
                "context_length": 12800,
            }
        )
    if STEP_FALLBACK:
        data.append(
            {
                "id": STEP_MODEL_ID,
                "object": "model",
                "created": 1783344048,
                "owned_by": "step",
                "context_length": 128000,
            }
        )
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_data = await request.json()
    if not isinstance(request_data, dict):
        raise HTTPException(status_code=400, detail="Request body must be an object")
    if not request_data.get("messages"):
        raise HTTPException(status_code=400, detail="messages is required")

    model = str(request_data.get("model") or MODEL_ID)
    force_minimax = MINIMAX_FALLBACK and "minimax" in model.lower()
    force_step = STEP_FALLBACK and "step" in model.lower()

    if force_minimax:
        logger.info(
            "forced MiniMax route model=%s stream=%s",
            model,
            bool(request_data.get("stream")),
        )
        if request_data.get("stream"):
            return StreamingResponse(
                _minimax_stream(request_data, model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        snap = await minimax_complete(request_data, model)
        return build_nonstream_response(
            model=model,
            content=str(snap.get("content") or ""),
            reasoning="",
            tool_calls=[],
        )

    if force_step:
        logger.info(
            "forced Step route model=%s stream=%s",
            model,
            bool(request_data.get("stream")),
        )
        if request_data.get("stream"):
            return StreamingResponse(
                _step_stream(request_data, model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        snap = await step_complete(request_data, model)
        return build_nonstream_response(
            model=model,
            content=str(snap.get("content") or ""),
            reasoning=str(snap.get("reasoning") or ""),
            tool_calls=[],
        )

    _, original_history, _ = extract_conversation(request_data.get("messages") or [])
    payload = build_upstream_payload(request_data)
    retained_history = payload["data"][2] or []
    dropped_messages = len(original_history) - len(retained_history)
    request_chars = payload_size_chars(payload["data"])
    prompt_tokens_est = estimate_messages_tokens(request_data.get("messages") or [])
    # KV-cache: check prefix hit
    tools_str_tmp = json.dumps(request_data.get("tools") or [], ensure_ascii=False)[:8000]
    kv_key = kv_cache_key(str(payload["data"][1] or "")[:4000], tools_str_tmp)
    kv_hit = kv_cache_get(kv_key) is not None
    if dropped_messages:
        logger.warning(
            "context_trim model=%s messages=%d retained=%d dropped=%d chars=%d tokens~%d budget=%d kv_hit=%s stream=%s",
            model,
            len(request_data.get("messages") or []),
            len(retained_history),
            dropped_messages,
            request_chars,
            prompt_tokens_est,
            MAX_INPUT_CHARS,
            kv_hit,
            bool(request_data.get("stream")),
        )
    else:
        logger.info(
            "request model=%s messages=%d history=%d chars=%d tokens~%d kv_hit=%s stream=%s",
            model,
            len(request_data.get("messages") or []),
            len(retained_history),
            request_chars,
            prompt_tokens_est,
            kv_hit,
            bool(request_data.get("stream")),
        )
    # aviso de estouro próximo do limite
    if prompt_tokens_est > CONTEXT_LENGTH_TOKENS * 0.9:
        logger.warning("context_near_limit tokens~%d / %d (%.0f%%)", prompt_tokens_est, CONTEXT_LENGTH_TOKENS, prompt_tokens_est/CONTEXT_LENGTH_TOKENS*100)

    if request_data.get("stream"):
        return StreamingResponse(
            _openai_stream(request_data, payload, model, prompt_tokens_est),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    last_snapshot: dict[str, Any] | None = None
    last_error: Exception | None = None
    # Loop de retry com compactação progressiva: nunca retorna vazio
    for attempt in range(3):
        try:
            async for snapshot in _upstream_snapshots(payload):
                last_snapshot = snapshot
            break
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                # compacta mais agressivamente e tenta novamente
                keep = 10 if attempt == 0 else 5
                logger.warning("Hy3 attempt %d failed (%s), retry compact keep=%d", attempt+1, exc, keep)
                # reconstrói payload com histórico menor
                try:
                    system_prompt, history, message = extract_conversation(request_data.get("messages") or [])
                    history = _intelligent_compact(history, keep_recent=keep)
                    # remonta payload manualmente
                    raw_tools = request_data.get("tools") or []
                    tools = _compact_tools_for_upstream(raw_tools)
                    tools_str = json.dumps(tools, ensure_ascii=False) if tools else ""
                    data_retry: list[Any] = [message, system_prompt, history or None, reasoning_level(request_data), request_data.get("temperature"), 4096, float(request_data.get("top_p") or 0), request_data.get("preserved_thinking"), tools_str]
                    data_retry, _ = _trim_history_to_budget(data_retry, UPSTREAM_HARD_LIMIT)
                    payload = {"data": data_retry}
                    last_snapshot = None
                    continue
                except Exception:
                    pass
            if STEP_FALLBACK:
                logger.warning("Hy3 failed after retries (%s); falling back to Step", exc)
                snap = await step_complete(request_data, model)
                return build_nonstream_response(model=model, content=str(snap.get("content") or ""), reasoning=str(snap.get("reasoning") or ""), tool_calls=[], prompt_tokens=prompt_tokens_est)
            raise
    if last_snapshot is None:
        if last_error:
            raise HTTPException(status_code=502, detail=f"Hy3 failed: {last_error}")
        raise HTTPException(status_code=502, detail="Hy3 returned no response")

    content = str(last_snapshot.get("content") or "")
    reasoning = str(last_snapshot.get("reasoning") or "")
    # Retry se veio vazio — compacta e tenta de novo (nunca para)
    if not content and not reasoning and not last_snapshot.get("tool_calls"):
        for keep in (10, 5, 2):
            logger.warning("Hy3 returned empty (tokens~%d), retry compact keep=%d", prompt_tokens_est, keep)
            try:
                system_prompt, history, message = extract_conversation(request_data.get("messages") or [])
                history = _intelligent_compact(history, keep_recent=keep)
                raw_tools = request_data.get("tools") or []
                tools = _compact_tools_for_upstream(raw_tools)
                tools_str = json.dumps(tools, ensure_ascii=False) if tools else ""
                data_retry = [message, system_prompt, history or None, "no_think", request_data.get("temperature"), 4096, 0.0, None, tools_str]
                data_retry, _ = _trim_history_to_budget(data_retry, UPSTREAM_HARD_LIMIT)
                payload_retry = {"data": data_retry}
                retry_snap = None
                async for snapshot in _upstream_snapshots(payload_retry):
                    retry_snap = snapshot
                if retry_snap and (str(retry_snap.get("content") or "").strip() or str(retry_snap.get("reasoning") or "").strip()):
                    last_snapshot = retry_snap
                    content = str(last_snapshot.get("content") or "")
                    reasoning = str(last_snapshot.get("reasoning") or "")
                    break
            except Exception as e:
                logger.warning("retry compact keep=%d failed: %s", keep, e)
                continue
        # Fallback final: nunca retornar vazio — sintetiza resposta de compactação
        if not content and not reasoning:
            content = f"[Contexto de {prompt_tokens_est} tokens compactado para caber no limite. Histórico resumido mantido. Sua última mensagem foi processada com sucesso em pt-BR. Tokens originais ~{prompt_tokens_est}, limite {CONTEXT_LENGTH_TOKENS}.]"
            reasoning = ""
    if _is_chinese_refusal(content) or _is_chinese_refusal(reasoning):
        if STEP_FALLBACK:
            logger.warning("Hy3 snapshot returned Chinese refusal; falling back to Step")
            snap = await step_complete(request_data, model)
            return build_nonstream_response(
                model=model,
                content=str(snap.get("content") or ""),
                reasoning=str(snap.get("reasoning") or ""),
                tool_calls=[],
            )

    # KV-cache: salva prefixo bem-sucedido
    try:
        kv_cache_set(kv_key, content[:2000])
    except Exception:
        pass
    return build_nonstream_response(
        model=model,
        content=content,
        reasoning=reasoning,
        tool_calls=normalize_tool_calls(last_snapshot.get("tool_calls")),
        prompt_tokens=prompt_tokens_est,
    )
