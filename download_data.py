from datasets import load_dataset

def download():
    print("Downloading dataset...")
    # Download to default cache
    load_dataset("sguajardo799/BinauralDenoising2025", split="train")
    print("Download complete.")

if __name__ == "__main__":
    download()
