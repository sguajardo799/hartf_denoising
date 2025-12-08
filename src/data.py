import torch
from torch.utils.data import Dataset
import torchaudio
import torch.nn.functional as F
from datasets import load_dataset
import os

class BinauralDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "sguajardo799/BinauralDenoising2025",
        split: str = "train",
        target_sample_rate: int = 16000,
        max_items: int | None = None,
        auth_token: str | None = None,
    ):
        super().__init__()
        self.target_sr = target_sample_rate
        
        # Load dataset from Hugging Face
        # Assuming the dataset has a 'split' column or we filter by it if it's a single split loaded
        # But usually HF datasets are loaded by split argument if defined in dataset script
        # The user mentioned a 'split' column in metadata.csv. 
        # We'll load the full dataset and filter, or load specific split if supported.
        # For now, let's load 'train' split of the HF dataset object, then filter by the 'split' column if needed.
        
        print(f"Loading dataset {dataset_name}...")
        
        # Strategy: Load metadata.csv directly to avoid AudioFolder overhead/errors
        # We construct the URL for metadata.csv
        # Assuming main branch
        base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
        metadata_url = f"{base_url}/metadata.csv"
        
        try:
            # Load CSV in streaming mode
            self.dataset = load_dataset("csv", data_files=metadata_url, split="train", streaming=True)
        except Exception as e:
             print(f"Error loading metadata from {metadata_url}: {e}")
             # Fallback to standard load if CSV fails (maybe it's not public or requires token in a way csv loader doesn't handle?)
             # Note: load_dataset("csv") might not use the auth_token by default for private repos unless passed?
             # It accepts use_auth_token (deprecated) or token?
             # Actually load_dataset("csv", ..., token=token) works.
             print("Falling back to standard load_dataset...")
             self.dataset = load_dataset(dataset_name, split="train", token=auth_token, streaming=True)

        # Filter by split
        if split:
             # We assume "split" column exists in CSV
             self.dataset = self.dataset.filter(lambda x: x["split"] == split)

        if max_items is not None:
            print(f"Materializing {max_items} items from stream...")
            self.dataset = list(self.dataset.take(max_items))
        else:
            # If not max_items, we might want to materialize everything or keep streaming?
            # For training, map-style is preferred.
            # But if dataset is huge, we might crash.
            # For now, let's materialize if max_items is None too, assuming it fits in RAM or user sets max_items.
            # Or we can keep it streaming but __getitem__ needs to handle it?
            # But __getitem__ uses idx.
            # So we MUST materialize to list to support __getitem__(idx).
            # If the dataset is huge, this code will OOM.
            # But the user asked for "few data samples" test.
            # For production, we should probably use standard load_dataset (not streaming) which uses Arrow (mmap) and doesn't OOM.
            # But standard load failed/was slow.
            # Let's assume for this task we materialize.
            print("Materializing full dataset from stream (warning: might be slow/OOM)...")
            self.dataset = list(self.dataset)

        self.base_url = base_url
        self.resampler = None 

    def __len__(self):
        return len(self.dataset)

    def _load_audio(self, source):
        # source is the path from CSV, e.g. "audio/hartf_front/..."
        # We need to construct full URL and load.
        # torchaudio.load supports URLs?
        # It depends on backend. Soundfile supports URLs.
        
        if isinstance(source, str):
            if source.startswith("http"):
                url = source
            else:
                url = f"{self.base_url}/{source}"
            
            # We need to handle authentication if the repo is private.
            # torchaudio.load doesn't easily handle headers for auth.
            # If public, it's fine.
            # If private, we might need to download using requests with token, then load.
            # Let's assume public for now or try torchaudio.load.
            # If it fails, we might need a helper.
            
            try:
                # Attempt direct load
                wav, sr = torchaudio.load(url)
            except Exception as e:
                # Fallback: download to tmp
                import requests
                import tempfile
                import shutil
                
                # print(f"Direct load failed ({e}), trying download...")
                try:
                    with requests.get(url, stream=True) as r:
                        r.raise_for_status()
                        # Create temp file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                            shutil.copyfileobj(r.raw, tmp)
                            tmp.flush()
                            # Load from temp file
                            # We need to seek to 0 if copyfileobj moved pointer? 
                            # copyfileobj moves it.
                            # But we are opening it again via torchaudio.load(tmp.name)
                            # On windows/some systems opening an open file is tricky.
                            # But delete=True means it's gone when closed.
                            # Let's use delete=False and manual cleanup to be safe across OS/containers
                            pass
                        
                    # Better approach for temp file
                    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                    os.close(fd)
                    try:
                        with requests.get(url, stream=True) as r:
                            r.raise_for_status()
                            with open(tmp_path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192): 
                                    f.write(chunk)
                        
                        wav, sr = torchaudio.load(tmp_path)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                            
                except Exception as dl_e:
                    print(f"Download failed for {url}: {dl_e}")
                    raise dl_e
                
        elif isinstance(source, dict) and "array" in source:
             # If we fell back to standard load_dataset, it returns dict
             wav = torch.from_numpy(source["array"]).float()
             sr = source["sampling_rate"]
        else:
            raise ValueError(f"Unknown audio source: {source}")

        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
            
        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            wav = self.resampler(wav)
        return wav

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # item is a dict from CSV row
        # keys: hartf_front, hartf_back, hrtf_clean, etc.
        
        wav_front = self._load_audio(item["hartf_front"])
        wav_back = self._load_audio(item["hartf_back"])
        
        if wav_front.shape[0] == 1: wav_front = wav_front.repeat(2, 1)
        if wav_back.shape[0] == 1: wav_back = wav_back.repeat(2, 1)
            
        noisy = torch.cat([wav_front, wav_back], dim=0) # (4, T)
        
        clean = self._load_audio(item["hrtf_clean"]) # (2, T)
        if clean.shape[0] == 1: clean = clean.repeat(2, 1)

        min_len = min(noisy.shape[-1], clean.shape[-1])
        noisy = noisy[..., :min_len]
        clean = clean[..., :min_len]

        return noisy, clean

def get_data_splits(config):
    # We instantiate the dataset twice, one for train, one for val
    # The class handles filtering by split
    
    train_ds = BinauralDataset(
        dataset_name=config.data.dataset_name,
        split="train",
        target_sample_rate=config.audio.sample_rate,
        max_items=config.data.max_items
    )
    
    val_ds = BinauralDataset(
        dataset_name=config.data.dataset_name,
        split="val", # User said "val/test" in split column
        target_sample_rate=config.audio.sample_rate,
        max_items=config.data.max_items
    )
    
    return train_ds, val_ds
