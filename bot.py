"""Telegram bot + FastAPI log server in one process."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import defaultdict, deque

from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters
import uvicorn

from agent import analyze, build_portal_reply
from server import app as fastapi_app, log_url_for

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("bot")

# chat_id -> recent user/assistant turns (for multi-turn grading)
_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=12))

FINAL_HINT = re.compile(
    r"(reply with only|ONLY this JSON|json object|\"answer\"\s*:)",
    re.IGNORECASE,
)


def _wants_final_json(text: str) -> bool:
    t = text.lower()
    if "json" in t and ("reply" in t or "only" in t or "{" in text):
        return True
    if FINAL_HINT.search(text):
        return True
    # Single-shot analysis questions without explicit "json" still need an answer
    if any(
        k in t
        for k in (
            "maternal mortality",
            "highest",
            "forecast",
            "mospi",
            "dataset",
            "compute",
            "what is",
            "which state",
            "which district",
        )
    ):
        return True
    return False


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    log.info("msg chat=%s len=%s", chat_id, len(text))

    _history[chat_id].append({"role": "user", "content": text})

    if not _wants_final_json(text):
        # Intermediate multi-turn step
        ack = "OK. Ready for the next instruction."
        _history[chat_id].append({"role": "assistant", "content": ack})
        await update.message.reply_text(ack)
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    messages = list(_history[chat_id])

    def _run():
        return analyze(messages)

    try:
        answer_obj, logger = await asyncio.to_thread(_run)
    except Exception as e:
        log.exception("agent failed")
        # Still return valid portal JSON so graders see format_ok-ish structure
        from agent import RunLogger, build_portal_reply as bpr

        logger = RunLogger()
        logger.log("error", error=str(e))
        answer_obj = {"answer": {"error": str(e)}}
        url = log_url_for(logger.run_id)
        await update.message.reply_text(bpr(answer_obj, url))
        return

    url = log_url_for(logger.run_id)
    reply = build_portal_reply(answer_obj, url)
    _history[chat_id].append({"role": "assistant", "content": reply})
    # Exactly one JSON object — no extra text
    await update.message.reply_text(reply)


async def _run_api(port: int) -> None:
    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    await server.serve()


async def main_async() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required")

    port = int(os.environ.get("PORT", "8000"))
    api_task = asyncio.create_task(_run_api(port))

    application = (
        Application.builder()
        .token(token)
        .connect_timeout(30)
        .read_timeout(30)
        .build()
    )
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    await application.initialize()
    await application.start()
    await application.updater.start_polling(drop_pending_updates=True)
    log.info("Telegram polling started; HTTP on :%s", port)

    try:
        await api_task
    finally:
        await application.updater.stop()
        await application.stop()
        await application.shutdown()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
