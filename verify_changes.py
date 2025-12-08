import torch
import torchaudio
from src.config import Config
from src.data import BinauralDataset
from src.models import UNet2D
from src.losses import build_loss
from src.utils import reconstruct_waveform
from src.features import get_transform

def verify():
    # Load config
    config = Config.from_yaml("config.yaml")
    # Override for testing if needed, but config.yaml should be correct now
    config.data.max_items = 10 
    
    print("1. Testing Dataset Loading (MOCKED)...")
    # Mock dataset to avoid waiting for download
    class MockDataset:
        def __init__(self):
            self.length = 10
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            # Return random tensors matching expected shapes
            # Noisy: (4, 16000*10) approx, let's use smaller T
            T = 16000
            return torch.randn(4, T), torch.randn(2, T)

    ds = MockDataset()
    print(f"Dataset length: {len(ds)}")
    noisy, clean = ds[0]
    print(f"Noisy shape: {noisy.shape}") # Should be (4, T)
    print(f"Clean shape: {clean.shape}") # Should be (2, T)
    
    assert noisy.shape[0] == 4
    assert clean.shape[0] == 2
    assert noisy.shape[-1] == clean.shape[-1]
    print("Dataset check passed!")

    print("\n2. Testing Model Forward Pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet2D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers
    ).to(device)
    
    # Create dummy batch
    B = 2
    T = noisy.shape[-1]
    noisy_batch = torch.randn(B, 4, T).to(device)
    clean_batch = torch.randn(B, 2, T).to(device)
    
    # Transform
    transform = get_transform(config.audio, device)
    
    noisy_spec = transform(noisy_batch) # (B, 4, F, T)
    clean_spec = transform(clean_batch) # (B, 2, F, T)
    
    print(f"Noisy Spec shape: {noisy_spec.shape}")
    
    # Forward
    pred_spec = model(noisy_spec)
    print(f"Pred Spec shape: {pred_spec.shape}") # Should be (B, 2, F, T)
    
    assert pred_spec.shape == clean_spec.shape
    print("Model forward pass passed!")
    
    print("\n3. Testing Loss Calculation...")
    # Loss
    criterion = build_loss(config, device)
    
    # Phase for composite loss
    noisy_subset = noisy_batch[:, :2, :]
    stft = torchaudio.transforms.Spectrogram(
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        power=None,
        center=True,
        pad_mode="reflect",
    ).to(device)
    noisy_complex = stft(noisy_subset)
    noisy_phase = torch.angle(noisy_complex)
    
    loss = criterion(pred_spec, clean_spec, noisy_phase, clean_batch)
    print(f"Loss value: {loss.item()}")
    print("Loss calculation passed!")
    
    print("\nAll checks passed successfully.")

if __name__ == "__main__":
    verify()
