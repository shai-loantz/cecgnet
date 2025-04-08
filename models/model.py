import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import summarize
from torch import Tensor, randn, Size, sigmoid, cat
from torch.optim import AdamW, Optimizer

from settings import ModelConfig
from utils.logger import setup_logger, logger
from utils.metrics import calculate_aggregate_metrics


class Model(LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.our_logger = setup_logger()
        self.accumulated_outputs: list[Tensor] = []
        self.accumulated_labels: list[Tensor] = []

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        self.our_logger.debug(f'Training step {batch_idx=}, {batch[0].shape=}')
        inputs, targets = batch
        loss, _, _ = self._run_batch([inputs, targets], 'train')
        return loss

    def _metrics_step(self, batch: list[Tensor], name: str) -> None:
        self.our_logger.debug(f'{name} step {batch[0].shape=}')
        if self.trainer.global_rank != 0:
            return

        _, outputs, targets = self._run_batch(batch, name)
        self.accumulated_outputs.append(outputs.detach())
        self.accumulated_labels.append(targets.detach())

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> None:
        self._metrics_step(batch, 'val')

    def test_step(self, batch: list[Tensor], batch_idx: int) -> None:
        self._metrics_step(batch, 'test')

    def _metrics_epoch_end(self, name: str) -> None:
        self.our_logger.debug(f'start _metrics_epoch_end')
        if self.trainer.global_rank != 0:
            return

        y_pred = cat(self.accumulated_outputs, dim=0)
        y = cat(self.accumulated_labels, dim=0)
        self.accumulated_outputs.clear()
        self.accumulated_labels.clear()

        metrics = calculate_aggregate_metrics(y_pred, y, self.config.threshold)
        metrics_dict = {f'{name}_{key}': value for key, value in metrics.items()}
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.our_logger.debug('end _metrics_epoch_end')

    def on_validation_epoch_end(self) -> None:
        self._metrics_epoch_end('val')

    def on_test_epoch_end(self) -> None:
        self._metrics_epoch_end('test')

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
        logger.info((probs > config.threshold).int())
