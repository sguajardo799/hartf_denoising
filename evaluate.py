import os
import torch
import torchaudio
import pandas as pd
from tqdm.auto import tqdm
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, PerceptualEvaluationSpeechQuality, ScaleInvariantSignalNoiseRatio, SignalNoiseRatio
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from src.config import Config
from src.data import get_data_splits
from src.models import UNet2D
from src.features import get_transform
from src.utils import reconstruct_waveform

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Denoising Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--max_items", type=int, help="Override max_items")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda)")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to evaluate on")
    return parser.parse_args()

def evaluate():
    args = parse_args()
    
    # 1. Load Config
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = Config.from_yaml(config_path)
    
    # Override config
    if args.max_items is not None:
        config.data.max_items = args.max_items
    if args.device is not None:
        config.general.device = args.device
        
    device = config.general.device
    print(f"Device: {device}")

    # 2. Setup Data
    # We instantiate BinauralDataset directly to allow split selection
    from src.data import BinauralDataset
    
    cache_dir = "data/audio_cache"
    
    val_ds = BinauralDataset(
        dataset_name=config.data.dataset_name,
        split=args.split,
        target_sample_rate=config.audio.sample_rate,
        max_items=config.data.max_items,
        cache_dir=cache_dir
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=1, # Evaluate one by one for metrics
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # 3. Load Model
    model = UNet2D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        kernel_size=config.model.kernel_size,
        use_batchnorm=config.model.use_batchnorm,
        dropout=config.model.dropout,
        final_activation=None,
    ).to(device)

    best_model_path = os.path.join(config.general.results_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
         raise FileNotFoundError(f"Model file not found at {best_model_path}")
    
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 4. Setup Metrics
    fs = config.audio.sample_rate
    stoi = ShortTimeObjectiveIntelligibility(fs, extended=False).to(device)
    pesq = PerceptualEvaluationSpeechQuality(fs, 'wb').to(device)
    si_sdr = ScaleInvariantSignalNoiseRatio().to(device)
    snr = SignalNoiseRatio().to(device)
    mse = MeanSquaredError().to(device)
    mae = MeanAbsoluteError().to(device)

    # 5. Setup Transform
    transform = get_transform(config.audio, device)
    
    # Pre-calculate Spectrogram transform for phase extraction if needed
    # We need the complex spectrogram of the noisy signal to get the phase.
    # If transform_type is spectrogram, we can get it from torchaudio.transforms.Spectrogram(power=None)
    # If transform_type is melspectrogram, we still need STFT phase.
    
    stft = torchaudio.transforms.Spectrogram(
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        power=None, # Complex
        center=True,
        pad_mode="reflect",
    ).to(device)

    results = []

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(tqdm(val_loader)):
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Extract Phase from Noisy (use first 2 channels as reference)
            # noisy: (B, 4, T)
            noisy_subset = noisy[:, :2, :] # (B, 2, T)
            noisy_complex = stft(noisy_subset) # (B, 2, F, T)
            noisy_phase = torch.angle(noisy_complex)

            # Forward Pass
            noisy_spec = transform(noisy) # (B, 4, F, T)
            pred_spec = model(noisy_spec) # (B, 2, F, T)

            # Reconstruct Waveform
            # reconstruct_waveform expects (B, C, F, T)
            pred_wav = reconstruct_waveform(pred_spec, noisy_phase, config.audio, device) # (B, 2, T)
            
            # Ensure lengths match (inverse transform might have slight diff)
            min_len = min(pred_wav.shape[-1], clean.shape[-1])
            pred_wav = pred_wav[..., :min_len]
            clean = clean[..., :min_len]
            noisy = noisy[..., :min_len]

            # Calculate Metrics
            # Metrics usually expect (B, T) or (B, C, T) depending on metric.
            # TorchMetrics audio metrics often expect (B, T).
            # We flatten B and C to treat each channel as an independent sample for metric calculation.
            B, C, T = pred_wav.shape
            pred_flat = pred_wav.reshape(B * C, T).contiguous()
            clean_flat = clean.reshape(B * C, T).contiguous()
            
            m_stoi = stoi(pred_flat, clean_flat)
            try:
                m_pesq = pesq(pred_flat, clean_flat)
            except Exception as e:
                # PESQ can fail on silence or very short signals
                m_pesq = float('nan')
            
            m_sisdr = si_sdr(pred_flat, clean_flat)
            m_snr = snr(pred_flat, clean_flat)
            m_mse = mse(pred_flat, clean_flat)
            m_mae = mae(pred_flat, clean_flat)

            results.append({
                "id": i,
                "stoi": m_stoi.item(),
                "pesq": m_pesq.item() if not isinstance(m_pesq, float) else m_pesq,
                "si_sdr": m_sisdr.item() if not torch.isnan(m_sisdr) else float('nan'),
                "snr": m_snr.item() if not torch.isnan(m_snr) else float('nan'),
                "mse": m_mse.item() if not torch.isnan(m_mse) else float('nan'),
                "mae": m_mae.item() if not torch.isnan(m_mae) else float('nan')
            })

    # 6. Save Report
    df = pd.DataFrame(results)
    report_path = os.path.join(config.general.results_dir, "evaluation_report.csv")
    df.to_csv(report_path, index=False)
    
    print("\nEvaluation Summary:")
    print(df.mean(numeric_only=True))
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    evaluate()
