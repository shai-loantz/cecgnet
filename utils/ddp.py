import os
import torch.distributed as dist


def is_main_proc():
    return int(os.environ.get("SLURM_PROCID", 0)) == 0


def should_log():
    return is_main_proc() or dist.is_initialized()
