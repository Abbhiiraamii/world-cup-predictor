"""
utils/logger.py
================
Single place to configure logging so every module reports consistently.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from config import LOG_DIR

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: str = "app.log", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger that writes to console and to a shared file.

    Args:
        name: Usually ``__name__`` of the calling module.
        log_file: File name (created under ``config.LOG_DIR``) to append to.
        level: Logging level, defaults to INFO.

    Returns:
        A ready-to-use ``logging.Logger`` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_path = Path(LOG_DIR) / log_file
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
