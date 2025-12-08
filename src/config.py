import yaml
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class GeneralConfig:
    device: str
    seed: int
    results_dir: str
    device_id: int = 0

@dataclass
class DataConfig:
    dataset_name: str = "sguajardo799/BinauralDenoising2025"
    max_items: int | None = None
    num_workers: int = 0
    pin_memory: bool = True
    download: bool = False
    # Optional fields for backward compatibility if needed, but better to be clean
    # root: str = ""

@dataclass
class AudioConfig:
    transform_type: str
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int

@dataclass
class ModelConfig:
    in_channels: int
    out_channels: int
    base_channels: int
    num_layers: int
    kernel_size: int
    use_batchnorm: bool
    dropout: float

@dataclass
class TrainingConfig:
    batch_size: int
    max_epochs: int
    learning_rate: float
    patience: int
    min_delta: float
    log_interval: int

@dataclass
class LossConfig:
    type: str = "mse" # "mse", "l1", "huber", "composite"
    spectral_type: str = "mse" # Used if type is composite
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.1
    delta: float = 0.1

@dataclass
class Config:
    general: GeneralConfig
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        
        general_cfg = GeneralConfig(**cfg_dict.get("general", {}))
        if general_cfg.device == "cuda":
            general_cfg.device = f"cuda:{general_cfg.device_id}"
            
        return cls(
            general=general_cfg,
            data=DataConfig(**cfg_dict.get("data", {})),
            audio=AudioConfig(**cfg_dict.get("audio", {})),
            model=ModelConfig(**cfg_dict.get("model", {})),
            training=TrainingConfig(**cfg_dict.get("training", {})),
            loss=LossConfig(**cfg_dict.get("loss", {})),
        )
