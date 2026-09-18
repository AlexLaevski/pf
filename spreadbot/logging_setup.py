"""Logging that is readable in a terminal and greppable in a file."""

from __future__ import annotations

import logging
import sys
from typing import Optional

FORMAT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"
DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", logfile: Optional[str] = None) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(FORMAT, DATEFMT))
    root.addHandler(stream)

    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        root.addHandler(file_handler)

    # The SDK and aiohttp are chatty at DEBUG.
    for noisy in ("websockets", "aiohttp", "lighter", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
