import os
import argparse
import torch
import torchaudio
import numpy as np
from tqdm.auto import tqdm
from src.config import Config
from src.models import UNet2D
from src.features import get_transform
from src.utils import reconstruct_waveform
from src.data import get_data_splits

def load_model(config, device):
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
    
    print(f"Loading model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    return model

def process_audio(waveform, model, transform, stft, config, device):
    # waveform: (1, 4, T) or (4, T) -> we expect (B, 4, T)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
        
    with torch.no_grad():
        noisy = waveform.to(device) # (B, 4, T)
        
        # Extract Phase (use first 2 channels)
        noisy_subset = noisy[:, :2, :] # (B, 2, T)
        noisy_complex = stft(noisy_subset) # (B, 2, F, T)
        noisy_phase = torch.angle(noisy_complex)
        
        # Forward Pass
        noisy_spec = transform(noisy) # (B, 4, F, T)
        pred_spec = model(noisy_spec) # (B, 2, F, T)
        
        # Reconstruct
        pred_wav = reconstruct_waveform(pred_spec, noisy_phase, config.audio, device) # (B, 2, T)
        
        # Match length
        min_len = min(pred_wav.shape[-1], waveform.shape[-1])
        pred_wav = pred_wav[..., :min_len]
        
        return pred_wav.cpu() # (B, 2, T)

def run_val_samples(args, config, device):
    print(f"Running inference on {args.n} validation samples...")
    
    # Setup Data
    # We instantiate BinauralDataset directly to allow split selection
    from src.data import BinauralDataset
    
    val_ds = BinauralDataset(
        dataset_name=config.data.dataset_name,
        split=args.split,
        target_sample_rate=config.audio.sample_rate,
        max_items=config.data.max_items
    )
    
    # Select n random samples
    indices = np.random.choice(len(val_ds), args.n, replace=False)
    
    # Setup Model & Transforms
    model = load_model(config, device)
    transform = get_transform(config.audio, device)
    stft = torchaudio.transforms.Spectrogram(
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        power=None,
        center=True,
        pad_mode="reflect",
    ).to(device)
    
    output_dir = os.path.join(config.general.results_dir, "inference_val")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, idx in enumerate(tqdm(indices)):
        noisy, clean = val_ds[idx]
        # noisy: (4, T), clean: (2, T)
        # Add batch dim
        noisy_in = noisy.unsqueeze(0) # (1, 4, T)
            
        denoised = process_audio(noisy_in, model, transform, stft, config, device) # (1, 2, T)
        denoised = denoised.squeeze(0) # (2, T)
        
        # Save
        sample_id = f"sample_{i}_{idx}"
        torchaudio.save(os.path.join(output_dir, f"{sample_id}_noisy.wav"), noisy, config.audio.sample_rate)
        torchaudio.save(os.path.join(output_dir, f"{sample_id}_clean.wav"), clean, config.audio.sample_rate)
        torchaudio.save(os.path.join(output_dir, f"{sample_id}_denoised.wav"), denoised, config.audio.sample_rate)
        
    print(f"Saved {args.n} samples to {output_dir}")

def run_folder(args, config, device):
    print(f"Processing folder: {args.input_folder}")
    
    if not os.path.exists(args.input_folder):
        raise FileNotFoundError(f"Input folder not found: {args.input_folder}")
        
    output_dir = args.output_folder or os.path.join(config.general.results_dir, "inference_folder")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Model & Transforms
    model = load_model(config, device)
    transform = get_transform(config.audio, device)
    stft = torchaudio.transforms.Spectrogram(
        n_fft=config.audio.n_fft,
        hop_length=config.audio.hop_length,
        power=None,
        center=True,
        pad_mode="reflect",
    ).to(device)
    
    files = [f for f in os.listdir(args.input_folder) if f.endswith(".wav")]
    
    for f in tqdm(files):
        path = os.path.join(args.input_folder, f)
        wav, sr = torchaudio.load(path)
        
        # Resample if needed
        if sr != config.audio.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, config.audio.sample_rate)
            wav = resampler(wav)
            
        # Check channels
        if wav.shape[0] != 4:
            print(f"Skipping {f}: Expected 4 channels, got {wav.shape[0]}")
            continue
            
        # Add batch dim
        wav = wav.unsqueeze(0) # (1, 4, T)
            
        denoised = process_audio(wav, model, transform, stft, config, device)
        denoised = denoised.squeeze(0) # (2, T)
        
        save_path = os.path.join(output_dir, f"denoised_{f}")
        torchaudio.save(save_path, denoised, config.audio.sample_rate)
        
    print(f"Processed {len(files)} files. Saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Inference script for denoising model")
    parser.add_argument("--max_items", type=int, help="Override max_items")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda)")
    parser.add_argument("--split", type=str, default="val", help="Dataset split (for val_samples)")
    
    subparsers = parser.add_subparsers(dest="mode", required=True)
    
    # Mode 1: Validation Samples
    parser_val = subparsers.add_parser("val_samples", help="Select n samples from validation set")
    parser_val.add_argument("--n", type=int, default=5, help="Number of samples to select")
    
    # Mode 2: Folder
    parser_folder = subparsers.add_parser("folder", help="Process a folder of wav files")
    parser_folder.add_argument("--input_folder", type=str, required=True, help="Path to input folder")
    parser_folder.add_argument("--output_folder", type=str, default=None, help="Path to output folder (optional)")
    
    args = parser.parse_args()
    
    # Load Config
    config = Config.from_yaml("config.yaml")
    
    # Override config
    if args.max_items is not None:
        config.data.max_items = args.max_items
    if args.device is not None:
        config.general.device = args.device
        
    device = config.general.device
    print(f"Using device: {device}")
    
    if args.mode == "val_samples":
        run_val_samples(args, config, device)
    elif args.mode == "folder":
        run_folder(args, config, device)

if __name__ == "__main__":
    main()
