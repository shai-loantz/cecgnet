import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch.distributed as dist

from utils.ddp import is_main_proc, should_log

BASE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
LOG_DIR = BASE_DIR / Path('..') / Path("logs") / datetime.now().strftime("%Y%m%d-%H%M")
os.makedirs(LOG_DIR, exist_ok=True)


def get_file_handler(log_level: int, formatter: logging.Formatter, rank: int) -> logging.FileHandler:
    level_name = logging.getLevelName(log_level).lower()
    handler = logging.FileHandler(LOG_DIR / f"{level_name}_{rank}.log")
    handler.setLevel(log_level)
    handler.setFormatter(formatter)
    return handler


def setup_logger():
    """Setup logger for each GPU process in DDP"""
    rank = dist.get_rank() if dist.is_initialized() else 0  # Get GPU rank

    logger = logging.getLogger("cecgnet")
    logger.setLevel(logging.DEBUG if should_log() else logging.CRITICAL)
    logger.propagate = False

    rank_prefix = f'[rank {rank}]' if dist.is_initialized() else 'pre-dist'
    formatter = logging.Formatter(f'%(rank)s %(asctime)s - %(name)s - %(levelname)s - %(message)s', defaults={'rank': rank_prefix})

    debug_handler = get_file_handler(logging.DEBUG, formatter, rank)
    info_file_handler = get_file_handler(logging.INFO, formatter, rank)
    error_file_handler = get_file_handler(logging.ERROR, formatter, rank)

    # stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO if is_main_proc() else logging.CRITICAL)  # Only rank 0 prints
    stdout_handler.addFilter(lambda record: record.levelno == logging.INFO)
    stdout_handler.setFormatter(formatter)

    # stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING if is_main_proc() else logging.CRITICAL)  # Only rank 0 prints
    stderr_handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(debug_handler)
    logger.addHandler(info_file_handler)
    logger.addHandler(error_file_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)

    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # Override all relevant loggers
    for lib in {
        "torch", "torchvision", "torchaudio",
        "lightning", "lightning.pytorch", "pytorch_lightning",
        "wandb", "requests", "tqdm", "pydantic",
    }:
        lib_logger = logging.getLogger(lib)
        lib_logger.handlers = logger.handlers  # Use the same handlers
        lib_logger.setLevel(logging.INFO)
        lib_logger.propagate = False  # Prevent duplicate logs

    logger.debug(f'Logger initialized for rank {rank}')
    return logger


logger = setup_logger()
