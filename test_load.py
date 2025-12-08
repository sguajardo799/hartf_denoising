from datasets import load_dataset
import os

try:
    print("Attempting to load dataset (non-streaming)...")
    ds = load_dataset("sguajardo799/BinauralDenoising2025", split="train")
    print(f"Dataset loaded. Length: {len(ds)}")
    print("Sample item keys:", ds[0].keys())
    print("Sample item:", ds[0])
except Exception as e:
    print(f"Error loading dataset: {e}")
