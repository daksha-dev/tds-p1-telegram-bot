"""HTTP server: health + public JSONL logs. Runs alongside the Telegram bot."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

app = FastAPI(title="tds-p1-telegram-bot")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/logs/{run_id}.jsonl")
def get_log(run_id: str):
    # Allow only safe run ids
    if not run_id or any(c in run_id for c in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="invalid run id")
    path = LOG_DIR / f"{run_id}.jsonl"
    if not path.is_file():
        # Also accept bare name without forcing extension twice
        alt = LOG_DIR / run_id
        if alt.suffix == ".jsonl" and alt.is_file():
            path = alt
        else:
            raise HTTPException(status_code=404, detail="log not found")
    return FileResponse(
        path,
        media_type="application/x-ndjson",
        filename=f"{run_id}.jsonl",
    )


@app.get("/")
def root() -> PlainTextResponse:
    return PlainTextResponse(
        "TDS P1 Telegram data-analyst bot. GET /health or /logs/<run_id>.jsonl\n"
    )


def public_base_url() -> str:
    return os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def log_url_for(run_id: str) -> str:
    return f"{public_base_url()}/logs/{run_id}.jsonl"
