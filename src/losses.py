import torch
import torch.nn as nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from src.utils import reconstruct_waveform

class CompositeLoss(nn.Module):
    def __init__(self, spectral_loss_type: str, alpha: float, beta: float, config, device: str):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.config = config
        self.device = device
        
        if spectral_loss_type == "mse":
            self.spectral_loss = nn.MSELoss()
        elif spectral_loss_type == "l1":
            self.spectral_loss = nn.L1Loss()
        elif spectral_loss_type == "huber":
            self.spectral_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown spectral loss type: {spectral_loss_type}")
            
        self.si_sdr = ScaleInvariantSignalNoiseRatio().to(device)

    def forward(self, pred_spec, clean_spec, noisy_phase, clean_wav):
        # 1. Spectral Loss
        spec_loss = self.spectral_loss(pred_spec, clean_spec)
        
        # 2. Time Domain Loss (SI-SDR)
        # Reconstruct waveform from predicted spectrogram and noisy phase
        # noisy_phase should be (B, C, F, T) matching pred_spec
        pred_wav = reconstruct_waveform(pred_spec, noisy_phase, self.config, self.device)
        
        # Ensure lengths match
        min_len = min(pred_wav.shape[-1], clean_wav.shape[-1])
        pred_wav = pred_wav[..., :min_len]
        clean_wav = clean_wav[..., :min_len]
        
        # SI-SDR is higher is better, so we minimize negative SI-SDR
        # SI-SDR expects (Batch, Time) or (Batch, Channels, Time) depending on implementation
        # torchmetrics ScaleInvariantSignalNoiseRatio usually expects (Batch, Time)
        # If we have (Batch, Channels, Time), we can flatten B*C or average.
        # Let's flatten B and C to treat each channel as an independent sample for SI-SDR
        
        B, C, T = pred_wav.shape
        pred_wav_flat = pred_wav.view(B * C, T)
        clean_wav_flat = clean_wav.view(B * C, T)
        
        sisdr_val = self.si_sdr(pred_wav_flat, clean_wav_flat)
        
        # Handle NaNs in SI-SDR (e.g. silence)
        if torch.isnan(sisdr_val):
             sisdr_val = torch.tensor(0.0, device=self.device)
        
        time_loss = -sisdr_val
        
        # Composite
        total_loss = self.alpha * spec_loss + self.beta * time_loss
        return total_loss

def build_loss(config, device: str):
    loss_type = config.loss.type
    
    if loss_type in ["mse", "l1", "huber"]:
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "huber":
            return nn.HuberLoss()
            
    elif loss_type == "composite":
        return CompositeLoss(
            spectral_loss_type=config.loss.spectral_type,
            alpha=config.loss.alpha,
            beta=config.loss.beta,
            config=config.audio,
            device=device
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
