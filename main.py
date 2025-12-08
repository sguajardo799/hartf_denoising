import os
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.data import get_data_splits
from src.models import UNet2D
from src.features import get_transform
from src.train import train_model
from src.losses import build_loss

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Denoising Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--max_epochs", type=int, help="Override max_epochs")
    parser.add_argument("--max_items", type=int, help="Override max_items")
    parser.add_argument("--batch_size", type=int, help="Override batch_size")
    parser.add_argument("--learning_rate", type=float, help="Override learning_rate")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda)")
    parser.add_argument("--download", action="store_true", help="Download entire dataset to cache")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Cargar configuración
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = Config.from_yaml(config_path)
    
    # Override config with args
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.max_items is not None:
        config.data.max_items = args.max_items
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.device:
        config.general.device = args.device
    if args.download:
        config.data.download = True

    print(f"Device: {config.general.device}")

    # Optimizations for A100
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("TF32 and CUDNN Benchmark enabled")

    # 2. Setup Data
    # Asegurar directorios (aunque HF maneja su cache)
    os.makedirs(config.general.results_dir, exist_ok=True)
    
    # get_data_splits ahora instancia BinauralDataset internamente
    train_ds, val_ds = get_data_splits(config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    # 3. Setup Model
    model = UNet2D(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        num_layers=config.model.num_layers,
        kernel_size=config.model.kernel_size,
        use_batchnorm=config.model.use_batchnorm,
        dropout=config.model.dropout,
        final_activation=None,
    ).to(config.general.device)

    # 4. Setup Training
    criterion = build_loss(config, config.general.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # 5. Setup Transform
    transform = get_transform(config.audio, config.general.device)

    # 6. Train
    train_model(config, model, train_loader, val_loader, transform, criterion, optimizer)

if __name__ == "__main__":
    main()
