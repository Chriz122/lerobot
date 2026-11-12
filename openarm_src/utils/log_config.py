from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "openarm.log"


def configure_logging() -> None:
    """Configure console and persistent OpenArm logging once per process."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if not any(getattr(handler, "_openarm_file_handler", False) for handler in root.handlers):
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler._openarm_file_handler = True
        root.addHandler(handler)

    if not any(getattr(handler, "_openarm_console_handler", False) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        handler._openarm_console_handler = True
        root.addHandler(handler)
