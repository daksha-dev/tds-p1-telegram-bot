# Manual deploy + portal steps (Q5)

## 1. Create Telegram bot

1. Open Telegram → `@BotFather`
2. `/newbot` → choose name → choose username ending in `bot`
3. Copy the **HTTP API token**

## 2. Rotate aipipe token

1. https://aipipe.org/login
2. Create/copy a **new** token (do not reuse any token pasted in chat)

## 3. Push is done via GitHub: `daksha-dev/tds-p1-telegram-bot`

## 4. Deploy on Render

1. https://dashboard.render.com → New → Web Service
2. Connect `daksha-dev/tds-p1-telegram-bot`
3. Runtime: Python
4. Build: `pip install -r requirements.txt`
5. Start: `python bot.py`
6. Health check path: `/health`
7. Environment:

| Key | Value |
|-----|--------|
| `TELEGRAM_BOT_TOKEN` | from BotFather |
| `AIPIPE_TOKEN` | from aipipe |
| `AIPIPE_BASE_URL` | `https://aipipe.org/openai/v1` |
| `AIPIPE_MODEL` | `gpt-4o-mini` |
| `PUBLIC_BASE_URL` | `https://<your-service>.onrender.com` (no trailing slash) |

8. Deploy → wait until healthy
9. Message your bot on Telegram with the MOSPI sample question
10. Confirm reply is pure JSON and `wget` the `log_url`

**Note:** Free Render services sleep after idle. Open `/health` or message the bot once before grading windows so it wakes up. Prefer a paid starter instance if grading is overnight.

## 5. Portal registration (0.1 marks)

Paste exactly (replace bot username):

```
https://github.com/daksha-dev/tds-p1-telegram-bot, YourBotUsername_bot
```

No `@` before the bot username unless the portal example shows it — use the bare username ending in `bot`.
