import torch
import torchaudio
import matplotlib.pyplot as plt
from src.config import AudioConfig



def plot_losses(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Evolución Train vs Val Loss (log-mel)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def reconstruct_waveform(log_spec: torch.Tensor, original_phase: torch.Tensor, config, device: str) -> torch.Tensor:
    """
    Reconstruct waveform from log-magnitude spectrogram using original phase.
    log_spec: (B, C, F, T) - Log-magnitude spectrogram (DB scale)
    original_phase: (B, C, F, T) - Phase of the noisy signal
    """
    # 1. DB to Amplitude
    # AmplitudeToDB uses 10 * log10(x) for power, or 20 * log10(x) for magnitude.
    # We used stype="magnitude", so x_db = 20 * log10(x).
    # x = 10^(x_db / 20)
    # Clamp log_spec to prevent overflow/underflow
    log_spec = torch.clamp(log_spec, min=-100.0, max=100.0)
    spec_mag = torch.pow(10.0, log_spec / 20.0)

    # 2. Combine with Phase
    # spec_complex = mag * e^(j*phase) = mag * (cos(phase) + j*sin(phase))
    # Note: For Mel, we first need to go back to Linear if possible, or use Griffin-Lim.
    # Here we assume we use the noisy phase for reconstruction which is standard for simple denoising.
    
    if config.transform_type == "spectrogram":
        spec_complex = torch.polar(spec_mag, original_phase)
        inverse_transform = torchaudio.transforms.InverseSpectrogram(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            center=True,
            pad_mode="reflect",
        ).to(device)
        waveform = inverse_transform(spec_complex)
        
    elif config.transform_type == "melspectrogram":
        # Inverse Mel Scale
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=config.n_fft // 2 + 1,
            n_mels=config.n_mels,
            sample_rate=config.sample_rate,
        ).to(device)
        
        spec_lin = inverse_mel(spec_mag)
        
        # Resize phase if needed (Mel reduces freq dimension) - actually phase is from STFT so it matches n_stft
        # spec_lin shape: (B, C, n_stft, T)
        # original_phase shape: (B, C, n_stft, T)
        
        spec_complex = torch.polar(spec_lin, original_phase)
        
        istft = torchaudio.transforms.InverseSpectrogram(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            center=True,
            pad_mode="reflect",
        ).to(device)
        waveform = istft(spec_complex)
        
    return waveform # (B, C, T)
