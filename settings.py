from enum import Enum
from typing import Optional, Any

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils.ddp import is_main_proc


class BaseConfig(BaseModel):
    """ Base class to allow easy pretraining overrides """

    def copy_with_override(self, **kwargs):
        filtered_kwargs = {k: v for k, v in kwargs.items() if hasattr(self, k)}
        return self.model_copy(update=filtered_kwargs)


class Attention(str, Enum):
    SEBlock = 'SE'
    SelfAttention = 'self_attention'
    SequentialAttention = 'sequential_attention'


class ModelName(str, Enum):
    RESNET = 'resnet'
    RESNET_ATTENTION = 'resnet_attention'
    VGG = 'vgg'
    SMALL = 'small'


class LightningStrategy(str, Enum):
    AUTO = 'auto'
    DDP = 'ddp'


class LightningAccelerator(str, Enum):
    GPU = 'gpu'
    CPU = 'cpu'


class DataConfig(BaseConfig):
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    input_length: int
    validation_size: float
    data_folder: Optional[str] = None
    test_data_folder: str
    positive_sampling_factor: float

    def get_data_loader_config(self) -> dict:
        return self.model_dump(exclude={"input_length", "validation_size", "positive_sampling_factor",
                                        "data_folder", "test_data_folder"})


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
    devices: list | int
    precision: str = "bf16-mixed"
    num_nodes: int = 1


class ModelConfig(BaseConfig):
    learning_rate: float
    weight_decay: float
    input_channels: int
    threshold: float
    use_weighted_loss: bool = True
    positive_prevalence: float
    warmup_steps: int = 1000

    attention: Attention = Attention.SelfAttention


class PreTrainConfig(BaseModel):
    data_folder: str
    max_epochs: int
    learning_rate: float
    weight_decay: float


class Config(BaseSettings):
    lightning: LightningConfig
    trainer: TrainerConfig
    data: DataConfig
    pre_process: PreprocessConfig
    model: ModelConfig
    model_folder: str = 'checkpoints'

    # pre training settings
    pretraining: bool
    pretraining_checkpoint_path: Optional[str] = None
    pre_trainer_config: PreTrainConfig

    pre_trainer: Optional[TrainerConfig] = None
    pre_model: Optional[ModelConfig] = None
    pre_data: Optional[DataConfig] = None

    model_name: ModelName = ModelName.RESNET
    checkpoint_name: Optional[str] = None
    manual_config: bool
    model_config = SettingsConfigDict(
        env_file=('config.env', 'config.local.env'),
        env_file_encoding='utf-8',
        env_nested_delimiter='__'
    )

    model_checkpoint_cb: Any = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.pretraining:
            pre_vars = self.pre_trainer_config.model_dump()
            # makes copies of the normal settings only changing the relevant parameters
            self.pre_model = self.model.copy_with_override(**pre_vars)
            self.pre_data = self.data.copy_with_override(**pre_vars)
            self.pre_trainer = self.trainer.copy_with_override(**pre_vars)

    def get_tester_params(self) -> dict:
        params = self.get_trainer_params()
        params.update({'devices': 1, 'strategy': 'auto'})
        return params

    def get_trainer_params(self, use_wandb: bool = True) -> dict:
        params = self.lightning.model_dump()
        params['logger'] = WandbLogger() if use_wandb and is_main_proc() else None
        self.model_checkpoint_cb = ModelCheckpoint(
            dirpath=self.model_folder,
            filename=self.get_checkpoint_name(),
            monitor="val_challenge_score",
            mode="max",
            save_top_k=1,
        )
        params['callbacks'] = [self.model_checkpoint_cb]
        params['enable_progress_bar'] = is_main_proc()

        if self.pretraining:
            params.update(self.pre_trainer.model_dump())
        else:
            params.update(self.trainer.model_dump())

        return params

    def get_checkpoint_name(self) -> str:
        name = self.checkpoint_name or self.model_name.value
        return f'pretraining_{name}' if self.pretraining else name

    def update_settings(self, data_folder: str, model_folder: str):
        self.data.data_folder = data_folder
        self.model_folder = model_folder

    def update_wandb_config(self, wandb_config: dict):
        for key, value in wandb_config.items():
            keys = key.split('__')
            if len(keys) == 1 and hasattr(self, key):
                setattr(self, key, value)
            elif len(keys) == 2 and hasattr(self, keys[0]):
                attr = getattr(self, keys[0])
                if hasattr(attr, keys[1]):
                    setattr(attr, keys[1], value)


    def get_wandb_params(self) -> dict:
        return {
            'trainer': self.get_trainer_params(),
            'data': self.data.model_dump(),
            'pre_process': self.pre_process.model_dump(),
            'model': self.model.model_dump(),
            'model_folder': self.model_folder,
            'pretraining': self.pretraining,
            'pretraining_checkpoint_path': self.pretraining_checkpoint_path,
            'pre_trainer_config': self.pre_trainer_config.model_dump(),
            'pre_trainer': self.pre_trainer.model_dump(),
            'pre_model': self.pre_model.model_dump(),
            'pre_data': self.pre_data.model_dump(),
            'model_name': self.model_name.value,
            'checkpoint_name': self.get_checkpoint_name(),
        }
