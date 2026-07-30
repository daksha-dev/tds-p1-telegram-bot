"""Telegram bot + FastAPI log server in one process."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from collections import defaultdict, deque

from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
import uvicorn

from agent import RunLogger, analyze, build_portal_reply
from server import app as fastapi_app, log_url_for

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("bot")

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


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        await update.message.reply_text(
            "Bot online. Send a data-analysis question; I reply with one JSON object."
        )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    log.info("msg chat=%s len=%s", chat_id, len(text))

    _history[chat_id].append({"role": "user", "content": text})

    if not _wants_final_json(text):
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
        logger = RunLogger()
        logger.log("error", error=str(e))
        answer_obj = {"answer": {"error": str(e)}}

    url = log_url_for(logger.run_id)
    reply = build_portal_reply(answer_obj, url)
    _history[chat_id].append({"role": "assistant", "content": reply})
    await update.message.reply_text(reply)


def _run_telegram() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required")

    async def _amain() -> None:
        application = (
            Application.builder()
            .token(token)
            .connect_timeout(60)
            .read_timeout(60)
            .write_timeout(60)
            .pool_timeout(60)
            .build()
        )
        application.add_handler(CommandHandler("start", on_start))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

        # Critical: clear any webhook so long-polling receives messages
        await application.bot.delete_webhook(drop_pending_updates=True)
        log.info("Webhook cleared; starting polling as @%s", (await application.bot.get_me()).username)

        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        # Keep this thread's event loop alive
        await asyncio.Event().wait()

    asyncio.run(_amain())


def main() -> None:
    if not os.environ.get("TELEGRAM_BOT_TOKEN"):
        raise SystemExit("TELEGRAM_BOT_TOKEN is required")
    if not (os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")):
        log.warning("AIPIPE_TOKEN missing — analysis replies will error until set")

    port = int(os.environ.get("PORT", "8000"))
    t = threading.Thread(target=_run_telegram, name="telegram-polling", daemon=True)
    t.start()
    log.info("HTTP listening on 0.0.0.0:%s", port)
    uvicorn.run(fastapi_app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
