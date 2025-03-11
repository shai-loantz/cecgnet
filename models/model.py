import torch.nn as nn
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import summarize
from torch import Tensor, randn, Size, sigmoid
from torch.optim import AdamW, Optimizer

from helper_code import compute_challenge_score, compute_auc, compute_accuracy, compute_f_measure
from settings import ModelConfig


class Model(LightningModule):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()
        self.threshold = config.threshold

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        loss, _ = self._run_batch([inputs, targets], calculate_metrics=False)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._run_batch(batch)
        self.log('val_loss', loss, sync_dist=True)
        val_metrics = {f'val_{key}': value for key, value in metrics.items()}
        self.log_dict(val_metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss, metrics = self._run_batch(batch)
        self.log('test_loss', loss, sync_dist=True)
        test_metrics = {f'test_{key}': value for key, value in metrics.items()}
        self.log_dict(test_metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _run_batch(self, batch: list[Tensor], calculate_metrics: bool = True) -> tuple[Tensor, dict[str, Tensor]]:
        inputs, targets = batch
        outputs = self(inputs)

        metrics = self._calculate_metrics(outputs.clone(), targets.clone()) if calculate_metrics else {}
        return self.criterion(outputs, targets), metrics

    def _calculate_metrics(self, y_pred: Tensor, y: Tensor) -> dict[str, Tensor]:
        labels = y.detach().cpu().float().numpy()
        prob_outputs = sigmoid(y_pred.detach()).cpu().float().numpy()
        binary_outputs = (prob_outputs > self.threshold).astype(int)
        challenge_score = compute_challenge_score(labels, prob_outputs)
        auroc, auprc = compute_auc(labels, prob_outputs)
        accuracy = compute_accuracy(labels, binary_outputs)
        f_measure = compute_f_measure(labels, binary_outputs)
        return {'challenge_score': challenge_score, 'auroc': auroc, 'auprc': auprc, 'accuracy': accuracy,
                'f_measure': f_measure}

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    @classmethod
    def test_model(cls) -> None:
        """Use this just to see the model structure has no errors"""
        batch_size = 7
        from settings import Config
        config = Config()
        x = randn(batch_size, config.model.input_channels, config.model.input_length)
        model = cls(config.model)
        print(model)
        print(summarize(model))
        logits = model(x).detach().squeeze()
        assert logits.size() == Size([batch_size])
        probs = sigmoid(logits)
        print(probs)
        print((probs > config.threshold).int())
