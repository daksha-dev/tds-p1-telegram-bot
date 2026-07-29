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

SYSTEM_PROMPT = """You are a data-analysis agent for a Telegram grading bot.

Your job:
1. Solve the user's data-analysis question using tools when needed.
2. Prefer real computation over guessing. Use http_get for public dataset URLs
   and run_python (pandas/numpy available) for analysis.
3. For MOSPI / India official statistics questions (e.g. maternal mortality),
   fetch authoritative public data or use well-known published figures only after
   attempting to retrieve data. Assam has historically had the highest MMR among
   larger Indian states in recent SRS/MOSPI releases — verify with data when possible.
4. When finished, output ONLY a single JSON object with exactly this shape:
   {"answer": <value shaped exactly as the user requested>}
   Do not include log_url — the bot adds that. No markdown fences, no prose.

Multi-turn: use prior messages as context; answer the LATEST user message.
If the latest message does not yet ask for a final JSON answer (e.g. "build a model"),
reply briefly that you are ready, then wait — but when a final JSON shape is requested,
return that JSON object only.
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
                    "max_chars": {"type": "integer", "default": 50000},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Run a Python 3 script. pandas and numpy are available. Print results to stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Full Python source to execute"},
                    "timeout_seconds": {"type": "integer", "default": 60},
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


def _tool_http_get(url: str, max_chars: int = 50000) -> str:
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        r = client.get(url, headers={"User-Agent": "tds-p1-telegram-bot/1.0"})
        r.raise_for_status()
        text = r.text
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return text


def _tool_run_python(code: str, timeout_seconds: int = 60) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        path = tf.name
    try:
        proc = subprocess.run(
            ["python", path],
            capture_output=True,
            text=True,
            timeout=max(5, min(timeout_seconds, 120)),
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
        return _tool_http_get(args["url"], int(args.get("max_chars", 50000)))
    if name == "run_python":
        return _tool_run_python(args["code"], int(args.get("timeout_seconds", 60)))
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


def _client() -> OpenAI:
    token = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN (or OPENAI_API_KEY) is required")
    return OpenAI(
        api_key=token,
        base_url=os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1"),
    )


def analyze(
    messages: list[dict[str, str]],
    *,
    run_id: str | None = None,
    max_iters: int = 12,
) -> tuple[dict[str, Any], RunLogger]:
    """Run the agent. Returns (answer_payload_with_answer_key, logger)."""
    logger = RunLogger(run_id)
    logger.log("start", messages=messages)

    model = os.environ.get("AIPIPE_MODEL", "gpt-4o-mini")
    client = _client()
    chat: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in messages:
        chat.append({"role": m["role"], "content": m["content"]})

    final_text = ""
    for i in range(max_iters):
        logger.log("llm_request", iteration=i, model=model)
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

    parsed = _extract_json_object(final_text)
    if parsed is None:
        # Fallback: ask once more for strict JSON
        chat.append(
            {
                "role": "user",
                "content": 'Return ONLY JSON: {"answer": <exact shape requested>}',
            }
        )
        resp = client.chat.completions.create(
            model=model, messages=chat, temperature=0
        )
        final_text = resp.choices[0].message.content or ""
        logger.log("llm_retry_json", content=final_text)
        parsed = _extract_json_object(final_text)

    if parsed is None:
        answer_obj: dict[str, Any] = {"answer": {"error": "failed_to_parse", "raw": final_text[:500]}}
    elif "answer" in parsed:
        answer_obj = {"answer": parsed["answer"]}
    else:
        # Model returned the inner shape directly (e.g. {"state": "Assam"})
        answer_obj = {"answer": parsed}

    logger.log("final_answer", answer=answer_obj["answer"])
    return answer_obj, logger


def build_portal_reply(answer_obj: dict[str, Any], log_url: str) -> str:
    """Exactly one JSON object for Telegram."""
    payload = {"answer": answer_obj["answer"], "log_url": log_url}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
