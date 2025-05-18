import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import summarize
from torch import Tensor, randn, Size, sigmoid, tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from data_tools.augmentations import get_augmentations
from settings import ModelConfig, AugmentationsConfig
from utils.logger import setup_logger, logger
from utils.metrics import calculate_metrics

METADATA_DIM = 2


class Model(LightningModule):
    def __init__(self, config: ModelConfig, augmentations: AugmentationsConfig) -> None:
        super().__init__()
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self._get_loss_weights())
        self.our_logger = None
        self.targets: list[Tensor] = []
        self.outputs: list[Tensor] = []
        self.augmentations = get_augmentations(augmentations, config.input_channels)

    def setup(self, stage=None):
        if self.our_logger is None:
            self.our_logger = setup_logger()
            self.our_logger.info(f"Logger initialized in setup() on rank {self.global_rank}")

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        inputs, targets, metadata = batch
        inputs = self.augmentations(inputs)
        loss, _, _ = self._run_batch([inputs, targets, metadata], 'train')
        return loss

    def _on_metric_epoch_start(self) -> None:
        self.targets = []
        self.outputs = []

    def _metric_step(self, batch: list[Tensor], step_name: str) -> None:
        _, outputs, targets = self._run_batch(batch, step_name)
        self.targets.append(targets.detach().cpu())
        self.outputs.append(outputs.detach().cpu())

    def _on_metric_epoch_end(self, step_name: str) -> None:
        targets = np.array(torch.cat(self.targets, dim=0).to(torch.float32)).flatten()
        outputs = np.array(torch.cat(self.outputs, dim=0).to(torch.float32)).flatten()
        metrics = calculate_metrics(targets, outputs, self.config.threshold)
        self.log_dict({f'{step_name}_{key}': value for key, value in metrics.items()})

    def on_validation_epoch_start(self) -> None:
        self._on_metric_epoch_start()

    def on_test_epoch_start(self) -> None:
        self._on_metric_epoch_start()

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> None:
        # _, outputs, targets = self._run_batch(batch, 'val')
        # if not self.trainer.sanity_checking and :
        #     logger.debug(f'Writing output for epoch {self.current_epoch}')
        #     write_outputs(self.trainer.global_rank, self.current_epoch, get_run_id(), outputs, targets)
        self._metric_step(batch, 'val')

    def test_step(self, batch: list[Tensor], batch_idx: int) -> None:
        self._metric_step(batch, 'test')

    def on_validation_epoch_end(self) -> None:
        self._on_metric_epoch_end('val')

    def on_test_epoch_end(self) -> None:
        self._on_metric_epoch_end('test')

    def _run_batch(self, batch: list[Tensor], name: str) -> tuple[Tensor, Tensor, Tensor]:
        inputs, targets, metadata = batch
        outputs = self(inputs, metadata)
        loss = self.criterion(outputs, targets)
        self.log(f'{name}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss, outputs, targets

    def configure_optimizers(self) -> dict:
        optimizer = AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

        def lr_lambda(current_step):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }

    def add_metadata(self, x: Tensor, metadata: tensor) -> Tensor:
        if x.shape[0] != metadata.shape[0]:
            raise ValueError("Batch size mismatch between x and metadata")
        if x.dim() > 2:
            raise ValueError("need 1D embeddings for metadata concatination")

        if metadata is None:
            metadata = torch.zeros(x.size(0), METADATA_DIM, device=x.device)
        elif torch.isnan(metadata).any():
            metadata = torch.nan_to_num(metadata, nan=0.0)

        # Concatenate along the feature dimension
        return torch.cat([x, metadata], dim=1)

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
            loss_ratio = (1 - p) / p
            logger.info(f'Initializing BCE loss with {loss_ratio=}')
            return tensor([loss_ratio])
        return None

    @classmethod
    def test_model(cls) -> None:
        """Use this just to see the model structure has no errors"""
        batch_size = 7
        from settings import Config
        config = Config()
        x = randn(batch_size, config.model.input_channels, config.data.input_length)
        model = cls(config.model, config.augmentations)
        logger.info(str(model))
        logger.info(summarize(model))
        logits = model(x).detach().squeeze()
        assert logits.size() == Size([batch_size])
        probs = sigmoid(logits)
        logger.info(probs)
        logger.info((probs > config.model.threshold).int())
