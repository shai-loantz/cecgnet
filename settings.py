from enum import Enum
from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Attention(str, Enum):
    SEBlock = 'SE'
    SelfAttention = 'self_attention'
    SequentialAttention = 'sequential_attention'


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
    input_length: int
    validation_size: float
    data_folder: Optional[str] = None

    def get_data_loader_config(self) -> dict:
        data_loader_config = self.model_dump(exclude={"input_length", "validation_size", "data_folder"})
        return data_loader_config


class PreprocessConfig(BaseModel):
    resample_freq: int
    low_cut_freq: float
    high_cut_freq: float


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
    threshold: float
    attention: Attention = Attention.SelfAttention


class Config(BaseSettings):
    lightning: LightningConfig
    trainer: TrainerConfig
    data_loader: DataLoaderConfig
    pre_process: PreprocessConfig
    model: ModelConfig
    model_folder: str = 'lightning_logs'

    # pre training settings
    pretraining: bool
    pre_trainer: Optional[TrainerConfig] = None
    pre_model: Optional[ModelConfig] = None
    pre_loader: Optional[DataLoaderConfig] = None

    model_name: ModelName = ModelName.SIMPLE
    checkpoint_name: Optional[str] = None
    manual_config: bool
    model_config = SettingsConfigDict(
        env_file='config.env',
        env_file_encoding='utf-8',
        env_nested_delimiter='__'
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.pretraining:
            # makes copies of the normal settings only changing the relevant parameters
            self.pre_model = self.model.model_copy(update={
                "learning_rate": self.pre_model.learning_rate,
                "weight_decay": self.pre_model.weight_decay,
            })
            self.pre_loader = self.data_loader.model_copy(update={
                "data_folder": self.pre_loader.data_folder,
            })
            self.pre_trainer = self.trainer.model_copy(update={
                "max_epochs": self.pre_trainer.max_epochs,
            })
        else:
            self.pre_model = None
            self.pre_loader = None
            self.pre_trainer = None

    def get_predictor_params(self) -> dict:
        params = self.lightning.model_dump()
        return params

    def get_trainer_params(self) -> dict:
        params = self.lightning.model_dump()
        params['logger'] = WandbLogger()
        params['callbacks'] = [ModelCheckpoint(
            dirpath=self.model_folder,
            filename=self.get_checkpoint_name(),  # why? just let it pick automatically, it might write over last one
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=True
        )]
        if self.pretraining:
            params.update(self.pre_trainer.model_dump())
        else:
            params.update(self.trainer.model_dump())
        return params

    def get_checkpoint_name(self) -> str:
        return self.checkpoint_name or self.model_name.value

    def update_settings(self, data_folder: str, model_folder: str):
        self.data_loader.data_folder = data_folder
        self.model_folder = model_folder
