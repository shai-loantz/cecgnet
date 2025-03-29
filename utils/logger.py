import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
now = datetime.now()
LOG_DIR = BASE_DIR / Path('..') / Path("logs") / now.strftime("%Y%m%d-%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("cecgnet")
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Debug file handler
debug_handler = logging.FileHandler(LOG_DIR / "debug.log")
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(formatter)

# Info file handler
info_file_handler = logging.FileHandler(LOG_DIR / "info.log")
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(formatter)

# Error file handler
error_file_handler = logging.FileHandler(LOG_DIR / "error.log")
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(formatter)

# stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
stdout_handler.setFormatter(formatter)

# stderr
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.WARNING)
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
