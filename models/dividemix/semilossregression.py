import numpy as np
import torch


class SemiLossRegression(torch.nn.Module):
    def __init__(self, rampup_length: int = 16):
        super().__init__()
        self.rampup_length = rampup_length

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, warmup, criterion, augments):
        device = outputs_x.device if outputs_x is not None else outputs_u.device
        # num_u = targets_u.size(0) if targets_u is not None else 0
        # num_x = targets_x.size(0) if targets_x is not None else 0
        # Lu_total = torch.tensor(0.0, device=device)
        # Lx_total = torch.tensor(0.0, device=device)

        if targets_u is not None and targets_u.numel() > 0:
            flat_out_u = outputs_u.flatten(0, 1)
            flat_tgt_u = targets_u.flatten(0, 1)
            Lu_total = criterion(flat_out_u, flat_tgt_u).sum()
        else:
            Lu_total = torch.tensor(0.0, device=device)

        if targets_x is not None and targets_x.numel() > 0:
            flat_out_x = outputs_x.flatten(0, 1)
            flat_tgt_x = targets_x.flatten(0, 1)
            Lx_total = criterion(flat_out_x, flat_tgt_x).sum()
        else:
            Lx_total = torch.tensor(0.0, device=device)
        # for i in range(augments):
        #     if num_u > 0:
        #         out_u = outputs_u[i * num_u: (i + 1) * num_u]
        #         Lu_total += criterion(out_u, targets_u)
        #
        #     if num_x > 0:
        #         out_x = outputs_x[i * num_x: (i + 1) * num_x]
        #         Lx_total += criterion(out_x, targets_x)

        weight = self.linear_rampup(warmup)
        return Lx_total, Lu_total, weight

    def linear_rampup(self, current):
        if current > 0:
            return 0
        current = np.clip((-current) / self.rampup_length, 0.0, 1.0)
        return float(current)
