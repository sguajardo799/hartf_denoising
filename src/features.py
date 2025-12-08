import torch
import torchaudio
from src.config import AudioConfig

import torch.nn as nn

def get_transform(config: AudioConfig, device: str):
    transforms = []
    
    if config.transform_type == "spectrogram":
        transforms.append(torchaudio.transforms.Spectrogram(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            power=1.0,
            center=True,
            pad_mode="reflect",
        ))
    elif config.transform_type == "melspectrogram":
        transforms.append(torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            center=True,
            pad_mode="reflect",
            power=1.0,
        ))
    else:
        raise ValueError(f"Unknown transform_type: {config.transform_type}")
    
    transforms.append(torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80))
    
    return nn.Sequential(*transforms).to(device)
