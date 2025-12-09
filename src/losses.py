import torch
import torch.nn as nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from asteroid.losses import SingleSrcPMSQE, SingleSrcNegSTOI
from src.utils import reconstruct_waveform

class CompositeLoss(nn.Module):
    def __init__(self, spectral_loss_type: str, alpha: float, beta: float, gamma: float, delta: float, config, device: str):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
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
        # Asteroid losses
        # sample_rate is required for PMSQE and STOI
        self.pmsqe = SingleSrcPMSQE(sample_rate=config.sample_rate).to(device)
        self.stoi = SingleSrcNegSTOI(sample_rate=config.sample_rate).to(device)

    def forward(self, pred_spec, clean_spec, noisy_phase, clean_wav):
        total_loss = 0.0

        # 1. Spectral Loss
        if self.alpha != 0.0:
            spec_loss = self.spectral_loss(pred_spec, clean_spec)
            total_loss += self.alpha * spec_loss
        
        # 2. Time Domain Losses (SI-SDR and STOI)
        if self.beta != 0.0 or self.delta != 0.0:
            # Reconstruct waveform from predicted spectrogram and noisy phase
            pred_wav = reconstruct_waveform(pred_spec, noisy_phase, self.config, self.device)
            
            # Ensure lengths match
            min_len = min(pred_wav.shape[-1], clean_wav.shape[-1])
            pred_wav = pred_wav[..., :min_len]
            clean_wav = clean_wav[..., :min_len]
            
            # Flatten B and C to treat each channel as an independent sample
            B, C, T = pred_wav.shape
            pred_wav_flat = pred_wav.reshape(B * C, T).contiguous()
            clean_wav_flat = clean_wav.reshape(B * C, T).contiguous()
            
            # SI-SDR (maximize -> minimize negative)
            if self.beta != 0.0:
                sisdr_val = self.si_sdr(pred_wav_flat, clean_wav_flat)
                if torch.isnan(sisdr_val): sisdr_val = torch.tensor(0.0, device=self.device)
                time_loss = -sisdr_val
                total_loss += self.beta * time_loss
            
            # STOI (minimize NegSTOI)
            if self.delta != 0.0:
                stoi_loss = self.stoi(pred_wav_flat, clean_wav_flat).mean()
                total_loss += self.delta * stoi_loss
        
        # PMSQE (minimize)
        if self.gamma != 0.0:
            # PMSQE expects spectrogram input (B, F, T)
            B_spec, C_spec, F_spec, T_spec = pred_spec.shape
            pred_spec_flat = pred_spec.reshape(B_spec * C_spec, F_spec, T_spec).contiguous()
            clean_spec_flat = clean_spec.reshape(B_spec * C_spec, F_spec, T_spec).contiguous()
            
            pmsqe_loss = self.pmsqe(pred_spec_flat, clean_spec_flat).mean()
            total_loss += self.gamma * pmsqe_loss
                      
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
            gamma=config.loss.gamma,
            delta=config.loss.delta,
            config=config.audio,
            device=device
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
