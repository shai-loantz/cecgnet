import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import torch.distributed as dist


def setup_logger():
    """Setup logger for each GPU process in DDP"""
    rank = dist.get_rank() if dist.is_initialized() else 0  # Get GPU rank
    base_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    now = datetime.now()
    log_dir = base_dir / Path('..') / Path("logs") / now.strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("cecgnet")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(f'[GPU {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Debug file handler
    debug_handler = logging.FileHandler(log_dir / f"debug_{rank}.log")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)

    # Info file handler
    info_file_handler = logging.FileHandler(log_dir / f"info_{rank}.log")
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(formatter)

    # Error file handler
    error_file_handler = logging.FileHandler(log_dir / f"error_{rank}.log")
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)

    # stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO if rank == 0 else logging.CRITICAL)  # Only rank 0 prints
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
    stdout_handler.setFormatter(formatter)

    # stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING if rank == 0 else logging.CRITICAL)  # Only rank 0 prints
    stderr_handler.setFormatter(formatter)

    if not logger.hasHandlers():
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

    return logger


logger = setup_logger()
