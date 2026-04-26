"""Structured logging setup."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(results_dir: str | Path, level_console: int = logging.INFO) -> None:
    """INFO+ to console; DEBUG+ to results/run.log."""
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    log_path = results_path / "run.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(level_console)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
