"""Logging setup for AML Monitoring project."""

import logging
import os
import sys
from pathlib import Path


def setup_logger(name="aml_monitoring", level=None, log_file=None):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = level or os.environ.get("LOG_LEVEL", "INFO")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
