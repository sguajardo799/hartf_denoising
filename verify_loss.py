import torch
from src.config import Config, LossConfig, AudioConfig
from src.losses import build_loss

def verify_loss():
    device = "cpu"
    
    # Mock Config
    loss_config = LossConfig(
        type="composite",
        spectral_type="l1",
        alpha=1.0,
        beta=0.1,
        gamma=0.1,
        delta=0.1
    )
    
    audio_config = AudioConfig(
        transform_type="spectrogram",
        sample_rate=16000,
        n_fft=512,
        hop_length=128,
        n_mels=64
    )
    
    class MockConfig:
        loss = loss_config
        audio = audio_config
        
    config = MockConfig()
    
    print("Building loss...")
    criterion = build_loss(config, device)
    print(f"Loss built: {criterion}")
    
    # Dummy Data
    B, C, F, T = 2, 2, 257, 100
    pred_spec = torch.randn(B, C, F, T, requires_grad=True).to(device)
    clean_spec = torch.randn(B, C, F, T).to(device)
    noisy_phase = torch.randn(B, C, F, T).to(device)
    
    # Waveform length corresponding to T=100 with hop=128
    # T = (L // hop) + 1 -> L approx (T-1)*hop
    L = (T - 1) * 128
    clean_wav = torch.randn(B, C, L).to(device)
    
    print("Forward pass...")
    loss = criterion(pred_spec, clean_spec, noisy_phase, clean_wav)
    print(f"Loss value: {loss.item()}")
    
    print("Backward pass...")
    loss.backward()
    print("Gradients computed.")
    
    if pred_spec.grad is not None:
        print("Gradient check passed.")
    else:
        print("Gradient check FAILED.")

if __name__ == "__main__":
    verify_loss()
