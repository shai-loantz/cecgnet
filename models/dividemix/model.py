import numpy as np
import torch
from lightning import LightningModule
from sklearn.mixture import GaussianMixture
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_tools.augmentations import get_augmentations
from models import MODELS
from models.dividemix.semilossregression import SemiLossRegression
from settings import Config
from utils.metrics import calculate_metrics

GMM_ITERATIONS = 10

"""
NOTE: manual_optimization must be True in trainer config
"""


class DivideMix(LightningModule):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.model1 = MODELS[config.model_name](config.model, config.augmentations)
        self.model2 = MODELS[config.model_name](config.model, config.augmentations)

        self.augmentations = get_augmentations(config.augmentations)
        self.warmup = config.divide_mix.warmup_epochs
        self.threshold = config.divide_mix.clean_threshold
        self.augments = config.divide_mix.augments
        self.lambda_u = config.divide_mix.lambda_unclean
        # these are here only for segmentation task, not relevant for this project
        self.regression = False
        self.temp = 0.5
        self.alpha = 0.75
        self.add_mixmatch = True
        self.num_class = 2  # potential classes for classification

        self.semi_loss = SemiLossRegression(config.divide_mix.rampup_length)
        self.losses = [[], []]
        self.criterion = torch.nn.MSELoss(reduction='none')

        self.dataL = None
        self.dataloader_config = config.data.get_data_loader_config()

    def training_step(self, batch: list[Tensor], batch_idx: int):
        loss1, loss2, _, _, _ = self._run_batch(batch, calculate_metrics=False)
        self.log('model1_train_loss', loss1.detach(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('model2_train_loss', loss2.detach(), on_epoch=True, prog_bar=True, sync_dist=True)
        self.backprop_step(loss1, loss2)

    def _on_metric_epoch_start(self):
        self.targets, self.outputs1, self.outputs2 = [], [], []

    def _metric_step(self, batch: list[Tensor], step_name: str) -> None:
        _, _, outputs1, outputs2, targets = self._run_batch(batch, calculate_metrics=True)
        self.targets.append(targets.detach().cpu())
        self.outputs1.append(outputs1.detach().cpu())
        self.outputs2.append(outputs2.detach().cpu())

    def _on_metric_epoch_end(self, step_name: str) -> None:
        targets = np.array(torch.cat(self.targets, dim=0).to(torch.float32)).flatten()
        outputs1 = np.array(torch.cat(self.outputs1, dim=0).to(torch.float32)).flatten()
        metrics1 = calculate_metrics(targets, outputs1, self.threshold)
        outputs2 = np.array(torch.cat(self.outputs2, dim=0).to(torch.float32)).flatten()
        metrics2 = calculate_metrics(targets, outputs2, self.threshold)
        self.log_dict({f'{step_name}_1_{key}': value for key, value in metrics1.items()})
        self.log_dict({f'{step_name}_2_{key}': value for key, value in metrics2.items()})

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


    def backprop_step(self, loss1: Tensor, loss2: Tensor):
        opt1, opt2 = self.optimizers()

        opt1.zero_grad()
        loss1.backward()
        opt1.step()

        opt2.zero_grad()
        loss2.backward()
        opt2.step()

    def configure_optimizers(self):
        opt1 = self.model1.configure_optimizers()
        opt2 = self.model2.configure_optimizers()
        return [opt1, opt2]

    def _eval_noise(self, model, loss_idx):
        model.eval()
        dtype = next(model.parameters()).dtype
        losses = []
        if self.dataL is None:
            self.dataL = DataLoader(self.trainer.datamodule.train_dataset.dataset, **self.dataloader_config)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataL, desc='Evaluating')):
                inputs, targets, metadata, *rest = batch  # safely unpacks first two items
                inputs = inputs.to(self.device).to(dtype)
                targets = targets.to(self.device)
                metadata = metadata.to(self.device)
                outputs = model(inputs, metadata)
                loss = self.criterion(outputs, targets)
                losses.append(loss.detach())

        losses = torch.cat(losses, dim=0).mean(dim=1)
        losses = (losses - losses.min()) / (losses.max() - losses.min()).clamp(min=1e-8)
        self.losses[loss_idx].append(losses)

        self.losses[loss_idx] = self.losses[loss_idx][-5:]
        input_loss = torch.stack(self.losses[loss_idx]).mean(0)
        input_loss = input_loss.reshape(-1, 1)

        prob = self.fit_probabilities(input_loss)
        return prob

    def fit_probabilities(self, input_loss: Tensor):
        if self.regression:
            # use a z score to imitate probabilities
            median = input_loss.median()
            mad = (input_loss - median).abs().median()
            mad = mad if mad > 1e-8 else torch.tensor(1e-8)  # handles all losses are equal
            z = 0.6745 * (input_loss - median) / mad
            prob = torch.sigmoid(-z)  # negative for lower z-scores = cleaner
        else:
            gmm = GaussianMixture(n_components=2, max_iter=GMM_ITERATIONS, tol=1e-2, reg_covar=5e-4)
            arr = input_loss.detach().cpu().numpy()
            gmm.fit(arr)
            prob = gmm.predict_proba(arr)
            prob = torch.tensor(prob[:, gmm.means_.argmin()], device=input_loss.device)
        return prob

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()  # needed for model checkpoint
        if self.warmup <= 1:
            self.prob1 = self._eval_noise(self.model1, 0)
            self.prob2 = self._eval_noise(self.model2, 1)
            self.trainer.datamodule.train_dataset.dataset.divide_mix(self.prob1.to('cpu'), self.prob2.to('cpu'))
        self.warmup -= 1

    def _refine_labels_u(self, inputs_u: Tensor, metadata_u:Tensor):
        """
        Label co-guessing for unlabeled inputs using both models.
        inputs_u: Tensor of shape (K, B, C, H, W) where K is number of augmentations.
        Returns: Refined targets_u of shape (B, num_classes)
        """
        self.model1.eval()
        self.model2.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(inputs_u.shape[0]):
                preds1 = self.model1(inputs_u[i], metadata_u[i])
                preds2 = self.model2(inputs_u[i], metadata_u[i])
                if not self.regression:
                    preds1 = torch.softmax(preds1, dim=1)
                    preds2 = torch.softmax(preds2, dim=1)
                all_preds.append(preds1)
                all_preds.append(preds2)
            # get average label from both models for unlabeled data
            predicted_u = torch.mean(torch.stack(all_preds), dim=0)
            if not self.regression:
                predicted_u = predicted_u ** (1 / self.temp)
                predicted_u = predicted_u / predicted_u.sum(dim=1, keepdim=True)
            targets_u = predicted_u.detach()
        return targets_u

    def _refine_labels_x(self, inputs_x, labels_x, w_x, secondary_model, metadata_x):
        """
        inputs_u: input unlabeled images of batch going through n augmentations (n, B, c, H, W)
        """
        secondary_model.eval()
        with torch.no_grad():
            preds = [secondary_model(inputs_x[i], metadata_x[i]) for i in range(inputs_x.shape[0])]
            if not self.regression:
                preds = [torch.softmax(preds[i], dim=1) for i in range(inputs_x.shape[0])]
            avg_preds = torch.mean(torch.stack(preds), dim=0)
            w_x = w_x.unsqueeze(1)
            predicted_x = w_x * labels_x + (1 - w_x) * avg_preds
            if not self.regression:
                predicted_x = predicted_x ** (1 / self.temp)
                predicted_x = predicted_x / predicted_x.sum(dim=1, keepdim=True)
            targets_x = predicted_x.detach()
        return targets_x

    def proc(self, xi, mi, yi=None, wi=None, sec = None):
        if xi.numel() == 0: return None, None, None
        aug = torch.stack([self.augmentations(xi) for _ in range(self.augments)])
        m_aug = mi.unsqueeze(0).expand(self.augments, -1, -1)
        labs = self._refine_labels_u(aug, m_aug) if yi is None else \
            self._refine_labels_x(aug, yi, wi, sec, m_aug)
        if labs.dim() == 1:
            labs = labs.unsqueeze(1)
        labs = labs.unsqueeze(0).repeat(self.augments, 1, 1)
        return aug, m_aug, labs

    def _run_batch_model(self, batch: list[Tensor], main_model, sec_model):
        """
        runs the batch on a single model (main model)
        batch: list of inputs, targets, probabilities for this model
        """
        main_model.train()
        sec_model.eval()
        inputs, targets, metadata, probabilities = batch
        mask = (probabilities >= self.threshold).squeeze()
        u = inputs[mask]
        metadata_u = metadata[mask]
        x = inputs[~mask]
        metadata_x = metadata[~mask]
        yx = targets[~mask]
        w_x = probabilities[~mask]

        # augment inputs and refine labels
        augment_u, metadata_u, labels_u = self.proc(u, metadata_u)
        augment_x, metadata_x, labels_x = self.proc(x, metadata_x, yx, w_x, sec_model)

        # combine inputs and labels
        inputs = torch.cat([i for i in [augment_u, augment_x] if i is not None], dim=1).flatten(0, 1)
        metadatas = torch.cat ([j for j in [metadata_u, metadata_x] if j is not None], dim=1).flatten(0, 1)
        if labels_u is not None and labels_x is not None:
            # (K, B_clean) + (K, B_noisy) → (K, B_clean + B_noisy)
            targets = torch.cat([labels_u, labels_x], dim=1)
        elif labels_u is not None:
            targets = labels_u
        else:
            targets = labels_x
        targets = targets.flatten(0, 1)
        # targets = torch.cat([t for t in [labels_u, labels_x] if t is not None], dim=0)
        if augment_u is not None:
            cutoff = augment_u.shape[0] * augment_u.shape[1]  # number of augments times batch size
        else:
            cutoff = 0

        if self.add_mixmatch:
            outputs, targets = self.mixmatch(inputs, targets, metadatas, main_model)
        else:
            outputs = main_model(inputs, metadatas)

        Lx, Lu, weight = self.semi_loss(outputs[cutoff:], targets[cutoff:], outputs[:cutoff],
                                        targets[:cutoff], self.warmup, main_model.criterion, self.augments)

        if self.regression:
            penalty = 0
        else:
            prior = torch.ones(self.num_class) / self.num_class
            prior = prior.cuda()
            pred_mean = torch.softmax(outputs, dim=1).mean(0)
            penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + weight * Lu * self.lambda_u + penalty
        return loss, outputs

    def mixmatch(self, inputs, targets, metadatas, main_model):
        idx = torch.randperm(inputs.shape[0], device=inputs.device)
        x2, y2, m2 = inputs[idx], targets[idx], metadatas[idx]


        l = np.random.beta(self.alpha, self.alpha)
        l = max(l, 1 - l)

        mixed_x = l * inputs + (1 - l) * x2
        mixed_y = l * targets + (1 - l) * y2
        mixed_m = m2

        outputs = main_model(mixed_x, mixed_m)
        targets = mixed_y
        return outputs, targets

    def _run_batch(self, batch: list[Tensor], calculate_metrics: bool = True):
        if self.warmup > 0 or calculate_metrics:
            self.model1.train()
            self.model2.train()
            inputs, targets, metadata, *rest = batch
            inputs = self.augmentations(inputs)
            outputs1 = self.model1(inputs, metadata)
            outputs2 = self.model2(inputs, metadata)
            loss1 = self.model1.criterion(outputs1, targets)
            loss2 = self.model2.criterion(outputs2, targets)
            return loss1, loss2, outputs1, outputs2, targets
        inputs, targets, metadata, prob1, prob2 = batch
        loss1, outputs1 = self._run_batch_model([inputs, targets, metadata, prob2], self.model1, self.model2)
        loss2, outputs2 = self._run_batch_model([inputs, targets, metadata, prob1], self.model2, self.model1)
        return loss1, loss2, outputs1, outputs2, targets
