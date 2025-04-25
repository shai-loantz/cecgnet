import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import summarize
from torch import Tensor, randn, Size, sigmoid, tensor
from torch.optim import AdamW, Optimizer

from settings import ModelConfig
from utils.logger import setup_logger, logger
from utils.metrics import write_outputs, calculate_metrics
from utils.run_id import get_run_id


class Model(LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self._get_loss_weights())
        self.our_logger = None

    def setup(self, stage=None):
        if self.our_logger is None:
            self.our_logger = setup_logger()
            self.our_logger.info(f"Logger initialized in setup() on rank {self.global_rank}")

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        self.our_logger.debug(f'Training step {batch_idx=}, {batch[0].shape=}')
        inputs, targets = batch
        loss, _, _ = self._run_batch([inputs, targets], 'train')
        return loss

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> None:
        self.our_logger.debug(f'val step {batch[0].shape=}')
        _, outputs, targets = self._run_batch(batch, 'val')
        if not self.trainer.sanity_checking:
            logger.debug(f'Writing output for epoch {self.current_epoch}')
            write_outputs(self.trainer.global_rank, self.current_epoch, get_run_id(), outputs, targets)

    def test_step(self, batch: list[Tensor], batch_idx: int) -> None:
        self.our_logger.debug(f'test step {batch[0].shape=}')
        _, outputs, targets = self._run_batch(batch, 'test')
        metrics = calculate_metrics(np.array(targets.to(torch.float32).cpu()),
                                    np.array(outputs.to(torch.float32).cpu()),
                                    self.config.threshold)
        self.log_dict({f'test_{key}': value for key, value in metrics.items()})

    def _run_batch(self, batch: list[Tensor], name: str) -> tuple[Tensor, Tensor, Tensor]:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log(f'{name}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss, outputs, targets

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def change_params(self, config: ModelConfig) -> None:
        """ used for changing pretraining to post training """
        self.config = config  # will also update optimizers when fit is called

    def _get_loss_weights(self) -> Tensor | None:
        """
        (1 -p)/p is negative_count/positive_count, and it computes the wanted weight for the positive samples.
        None means equal weight for positive and negative (normal loss).
        """
        if self.config.use_weighted_loss:
            p = self.config.positive_prevalence
            return tensor([(1 - p) / p])
        return None

    @classmethod
    def test_model(cls) -> None:
        """Use this just to see the model structure has no errors"""
        batch_size = 7
        from settings import Config
        config = Config()
        x = randn(batch_size, config.model.input_channels, config.data.input_length)
        model = cls(config.model)
        logger.info(str(model))
        logger.info(summarize(model))
        logits = model(x).detach().squeeze()
        assert logits.size() == Size([batch_size])
        probs = sigmoid(logits)
        logger.info(probs)
        logger.info((probs > config.model.threshold).int())
