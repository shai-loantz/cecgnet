from enum import Enum
from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelName(str, Enum):
    SIMPLE = 'simple'
    RESNET_ATTENTION = 'resnet_attention'


class LightningStrategy(str, Enum):
    AUTO = 'auto'
    DDP = 'ddp'


class LightningAccelerator(str, Enum):
    GPU = 'gpu'
    CPU = 'cpu'


class DataLoaderConfig(BaseModel):
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


class TrainerConfig(BaseModel):
    max_epochs: int = 30
    accumulate_grad_batches: int = 16


class LightningConfig(BaseModel):
    accelerator: LightningAccelerator = LightningAccelerator.GPU
    strategy: LightningStrategy = LightningStrategy.AUTO
    devices: str = "auto"
    precision: str = "bf16-mixed"
    num_nodes: int = 1


class ModelConfig(BaseModel):
    learning_rate: float
    weight_decay: float
    input_channels: int
    input_length: int
    threshold: float


class Config(BaseSettings):
    lightning: LightningConfig
    trainer: TrainerConfig
    data_loader: DataLoaderConfig
    model: ModelConfig

    model_name: ModelName = ModelName.SIMPLE
    checkpoint_name: Optional[str] = None
    validation_size: float
    manual_config: bool
    model_config = SettingsConfigDict(
        env_file='config.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__'
    )

    def get_predictor_params(self) -> dict:
        params = self.lightning.model_dump()
        return params

    def get_trainer_params(self, model_folder: str) -> dict:
        params = self.lightning.model_dump()
        params.update(self.trainer.model_dump())
        params['logger'] = WandbLogger()
        params['callbacks'] = [ModelCheckpoint(
            dirpath=model_folder,
            filename=self.get_checkpoint_name(),
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True
        )]
        return params

    def get_checkpoint_name(self) -> str:
        return self.checkpoint_name or self.model_name.value
