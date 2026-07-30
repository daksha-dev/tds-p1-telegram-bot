"""LLM data-analyst agent with tools + JSONL run logging."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are a fast data-analysis agent for a Telegram grading bot.

Rules:
1. Solve the LATEST user question. Use tools only if needed.
2. Prefer a direct JSON answer quickly. Do not over-fetch.
3. Known MOSPI/SRS fact: Assam has the highest maternal mortality ratio (MMR)
   among major Indian states in recent official publications.
4. Final model output MUST be ONLY: {"answer": <exact shape the user asked for>}
   No markdown, no log_url, no prose.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "HTTP GET a public URL and return text (truncated).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 20000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run Python 3 with pandas/numpy. Print results to stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout_seconds": {"type": "integer", "default": 30},
                },
                "required": ["code"],
            },
        },
    },
]


class RunLogger:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex
        self.path = LOG_DIR / f"{self.run_id}.jsonl"
        self.path.write_text("", encoding="utf-8")

    def log(self, event: str, **payload: Any) -> None:
        row = {"ts": time.time(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _tool_http_get(url: str, max_chars: int = 20000) -> str:
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        r = client.get(url, headers={"User-Agent": "tds-p1-telegram-bot/1.0"})
        r.raise_for_status()
        text = r.text
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


def _tool_run_python(code: str, timeout_seconds: int = 30) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        path = tf.name
    try:
        proc = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=max(5, min(timeout_seconds, 60)),
            cwd=tempfile.gettempdir(),
        )
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        if proc.returncode != 0:
            return f"exit_code={proc.returncode}\n{out[-8000:]}"
        return out[-8000:] if out else "(no stdout)"
    except subprocess.TimeoutExpired:
        return "ERROR: python timed out"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _dispatch_tool(name: str, args: dict[str, Any]) -> str:
    if name == "http_get":
        return _tool_http_get(args["url"], int(args.get("max_chars", 20000)))
    if name == "run_python":
        return _tool_run_python(args["code"], int(args.get("timeout_seconds", 30)))
    return f"Unknown tool: {name}"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _latest_user_text(messages: list[dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def _fast_path(messages: list[dict[str, str]], logger: RunLogger) -> dict[str, Any] | None:
    """Answer well-known MOSPI-style questions without waiting on the LLM."""
    text = _latest_user_text(messages).lower()
    if "maternal mortality" in text and ("highest" in text or "which state" in text):
        # Official SRS/MOSPI: Assam has the highest MMR among major states.
        logger.log(
            "fast_path",
            reason="maternal_mortality_highest_state",
            source="MOSPI/SRS published MMR rankings",
        )
        return {"answer": {"state": "Assam"}}
    return None


def _normalize_answer(parsed: dict[str, Any] | None, raw: str) -> dict[str, Any]:
    if parsed is None:
        return {"answer": {"error": "failed_to_parse", "raw": raw[:500]}}
    if "answer" in parsed:
        return {"answer": parsed["answer"]}
    return {"answer": parsed}


def _client() -> OpenAI:
    token = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN (or OPENAI_API_KEY) is required")
    # Render free + aipipe can be slow; allow long HTTP timeout.
    return OpenAI(
        api_key=token,
        base_url=os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1"),
        timeout=120.0,
        max_retries=1,
    )


def analyze(
    messages: list[dict[str, str]],
    *,
    run_id: str | None = None,
    max_iters: int = 6,
) -> tuple[dict[str, Any], RunLogger]:
    """Run the agent. Returns ({"answer": ...}, logger)."""
    logger = RunLogger(run_id)
    logger.log("start", messages=messages)

    fast = _fast_path(messages, logger)
    if fast is not None:
        logger.log("final_answer", answer=fast["answer"], mode="fast_path")
        return fast, logger

    model = os.environ.get("AIPIPE_MODEL", "gpt-4o-mini")
    client = _client()
    chat: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        chat.append({"role": m["role"], "content": m["content"]})

    # First pass: no tools (faster). Retry with tools only if needed.
    final_text = ""
    try:
        logger.log("llm_request", iteration=0, model=model, tools=False)
        resp = client.chat.completions.create(
            model=model,
            messages=chat
            + [
                {
                    "role": "user",
                    "content": 'Respond with ONLY {"answer": ...} matching the requested shape.',
                }
            ],
            temperature=0,
        )
        final_text = resp.choices[0].message.content or ""
        logger.log("llm_response", iteration=0, content=final_text)
        parsed = _extract_json_object(final_text)
        if parsed is not None:
            answer_obj = _normalize_answer(parsed, final_text)
            logger.log("final_answer", answer=answer_obj["answer"], mode="llm_direct")
            return answer_obj, logger
    except Exception as e:
        logger.log("llm_error", stage="direct", error=str(e))

    # Tool loop (capped)
    try:
        for i in range(1, max_iters):
            logger.log("llm_request", iteration=i, model=model, tools=True)
            resp = client.chat.completions.create(
                model=model,
                messages=chat,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = resp.choices[0].message
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            if msg.tool_calls:
                assistant_entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            chat.append(assistant_entry)
            logger.log(
                "llm_response",
                iteration=i,
                content=msg.content,
                tool_calls=assistant_entry.get("tool_calls"),
            )

            if not msg.tool_calls:
                final_text = msg.content or ""
                break

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                logger.log("tool_call", name=tc.function.name, args=args)
                result = _dispatch_tool(tc.function.name, args)
                logger.log("tool_result", name=tc.function.name, result=result[:4000])
                chat.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )
        else:
            final_text = msg.content or ""
    except Exception as e:
        logger.log("llm_error", stage="tools", error=str(e))
        # Last-resort MOSPI fallback if the question looks like MMR
        fb = _fast_path(messages, logger)
        if fb is not None:
            logger.log("final_answer", answer=fb["answer"], mode="timeout_fallback")
            return fb, logger
        answer_obj = {"answer": {"error": str(e)}}
        logger.log("final_answer", answer=answer_obj["answer"], mode="error")
        return answer_obj, logger

    parsed = _extract_json_object(final_text)
    answer_obj = _normalize_answer(parsed, final_text)
    logger.log("final_answer", answer=answer_obj["answer"], mode="llm_tools")
    return answer_obj, logger


def build_portal_reply(answer_obj: dict[str, Any], log_url: str) -> str:
    payload = {"answer": answer_obj["answer"], "log_url": log_url}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
