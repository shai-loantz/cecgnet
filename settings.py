from enum import Enum
from typing import Optional

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseConfig(BaseModel):
    """ Base class to allow easy pretraining overrides """

    def copy_with_override(self, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if not hasattr(self, k)}
        return self.model_copy(update=filtered_kwargs)


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


class DataLoaderConfig(BaseConfig):
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


class TrainerConfig(BaseConfig):
    max_epochs: int = 30
    accumulate_grad_batches: int = 16


class LightningConfig(BaseModel):
    accelerator: LightningAccelerator = LightningAccelerator.GPU
    strategy: LightningStrategy = LightningStrategy.AUTO
    devices: str = "auto"
    precision: str = "bf16-mixed"
    num_nodes: int = 1


class ModelConfig(BaseConfig):
    learning_rate: float
    weight_decay: float
    input_channels: int
    threshold: float
    attention: Attention = Attention.SelfAttention


class PreTrainerConfig(TrainerConfig):
    """ Pretraining-specific trainer config """
    pass


class PreModelConfig(ModelConfig):
    """ Pretraining-specific model config """
    pass


class PreDataLoaderConfig(DataLoaderConfig):
    """ Pretraining-specific data loader config """
    pass


class Config(BaseSettings):
    lightning: LightningConfig
    trainer: TrainerConfig
    data_loader: DataLoaderConfig
    pre_process: PreprocessConfig
    model: ModelConfig
    model_folder: str = 'lightning_logs'

    # pre training settings
    pretraining: bool
    pre_trainer: Optional[PreTrainerConfig] = None
    pre_model: Optional[PreModelConfig] = None
    pre_loader: Optional[PreDataLoaderConfig] = None
    pretraining_checkpoint_path: Optional[str] = None

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
            self.pre_model = PreModelConfig(**self.model.model_dump())
            self.pre_loader = PreDataLoaderConfig(**self.data_loader.model_dump())
            self.pre_trainer = PreTrainerConfig(**self.trainer.model_dump())
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
            filename=self.get_checkpoint_name(),
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
        if self.pretraining:
            return f'pretraining_{self.checkpoint_name or self.model_name.value}'
        return self.checkpoint_name or self.model_name.value

    def update_settings(self, data_folder: str, model_folder: str):
        self.data_loader.data_folder = data_folder
        self.model_folder = model_folder
