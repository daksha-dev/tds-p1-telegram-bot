"""Local CLI smoke test for the analyst agent (no Telegram)."""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agent import analyze, build_portal_reply
from server import log_url_for


def main() -> None:
    q = (
        sys.argv[1]
        if len(sys.argv) > 1
        else (
            "Which state has the highest maternal mortality rate based on MOSPI data? "
            'Reply with ONLY this JSON object and nothing else: '
            '{"answer": {"state": "<state name>"}, "log_url": "<public wget-able URL>"}'
        )
    )
    answer_obj, logger = analyze([{"role": "user", "content": q}])
    url = log_url_for(logger.run_id)
    print(build_portal_reply(answer_obj, url))
    print("LOG_FILE=", logger.path, file=sys.stderr)


if __name__ == "__main__":
    main()
