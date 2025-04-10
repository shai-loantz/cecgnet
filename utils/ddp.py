import os
import torch.distributed as dist

from utils.logger import logger


def is_main_proc():
    return int(os.environ.get("SLURM_PROCID", 0)) == 0


def should_log():
    return is_main_proc() or dist.is_initialized()


def get_run_id() -> str:
    try:
        with open('RUN_ID', 'r') as fh:
            return fh.read()
    except Exception:
        logger.exception('Could not get run id. Using placeholder')
        return 'RUN_ID_PLACEHOLDER'


def set_run_id(run_id: str) -> None:
    with open('RUN_ID', 'w') as fh:
        fh.write(run_id)
