# TDS P1 — Data Analyst Telegram Bot

LLM agent Telegram bot for IITM Tools in Data Science Project 1 (Q5).

When messaged a data-analysis question, the bot replies with **exactly one JSON object**:

```json
{"answer": <shape requested by the question>, "log_url": "https://<host>/logs/<run_id>.jsonl"}
```

- `answer` — graded result  
- `log_url` — public wget-able JSONL agent run log  

Grading pipeline (for local testing): https://github.com/Jivraj-18/tds-p1-t2-2026-telegram-bot

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) → `/newbot` (username must end in `bot`).
2. Get an [aipipe.org](https://aipipe.org/login) token.
3. Copy `.env.example` → `.env` and fill:

```
TELEGRAM_BOT_TOKEN=...
AIPIPE_TOKEN=...
PUBLIC_BASE_URL=https://your-service.onrender.com
```

4. Install and run locally:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python bot.py
```

CLI agent test (no Telegram):

```bash
python cli_test.py
```

## Deploy (Render)

1. Push this repo to GitHub (public).
2. Render → New → Web Service → connect repo.
3. Or use Blueprint: `render.yaml`.
4. Set env vars: `TELEGRAM_BOT_TOKEN`, `AIPIPE_TOKEN`, `PUBLIC_BASE_URL` (your Render URL, no trailing slash).
5. Health check: `/health`.
6. After deploy, message the bot on Telegram with a sample MOSPI question.

## Portal registration

Submit (comma-separated):

```
https://github.com/daksha-dev/tds-p1-telegram-bot, YourBotUsername_bot
```

## Endpoints

| Path | Purpose |
|------|---------|
| `GET /health` | Liveness |
| `GET /logs/<run_id>.jsonl` | Public run log |

## License

MIT
