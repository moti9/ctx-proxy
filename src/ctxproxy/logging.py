"""Structured logging.

Reduction decisions are the thing you will actually want to debug at 2am, so
they get a dedicated event logger that emits one line per decision with the
before/after token counts.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_EVENT_KEY = "_ctxproxy_event"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        event = getattr(record, _EVENT_KEY, None)
        if event:
            payload.update(event)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname[0]} {record.getMessage()}"
        event = getattr(record, _EVENT_KEY, None)
        if event:
            extras = " ".join(f"{k}={v}" for k, v in event.items())
            base = f"{base}  {extras}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure(level: str = "info", fmt: str = "text") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # uvicorn installs its own handlers; route them through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers[:] = []
        lg.propagate = True

    # httpx logs every request at INFO, which drowns out our own events.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def log_event(
    logger: logging.Logger, msg: str, /, level: int = logging.INFO, **fields: Any
) -> None:
    logger.log(level, msg, extra={_EVENT_KEY: fields})
