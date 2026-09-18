"""Structured, timestamped logging to a rotating file + stdout.

Every fetch, de-vig result, submission/response, and skip-with-reason is logged
so the full run is auditable after the fact.
"""

from __future__ import annotations

import logging
import logging.handlers

from . import config

_CONFIGURED = False


def get_logger(name: str = "jumpbot") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        fh = logging.handlers.RotatingFileHandler(
            config.LOG_DIR / "jumpbot.log", maxBytes=10_000_000, backupCount=5
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        root = logging.getLogger("jumpbot")
        root.addHandler(fh)
        root.addHandler(ch)
        root.setLevel(logging.DEBUG)
        root.propagate = False
        _CONFIGURED = True
    return logger
