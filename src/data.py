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
        cache_dir: str | None = None,
        download: bool = False,
    ):
        super().__init__()
        self.target_sr = target_sample_rate
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"Loading dataset {dataset_name}...")
        
        base_url = f"https://huggingface.co/datasets/{dataset_name}/resolve/main"
        metadata_url = f"{base_url}/metadata.csv"
        
        try:
            # Load CSV. For full download, we don't need streaming if metadata is small.
            # But let's keep streaming=False to get full list easily.
            self.dataset = load_dataset("csv", data_files=metadata_url, split="train", streaming=False)
        except Exception as e:
             print(f"Error loading metadata from {metadata_url}: {e}")
             print("Falling back to standard load_dataset...")
             self.dataset = load_dataset(dataset_name, split="train", token=auth_token, streaming=False)

        # Filter by split
        if split:
             self.dataset = self.dataset.filter(lambda x: x["split"] == split)

        if max_items is not None:
            print(f"Selecting {max_items} items...")
            self.dataset = self.dataset.select(range(max_items))

        self.base_url = base_url
        self.resampler = None 
        
        self.base_url = base_url
        self.resampler = None 
        
        if download and self.cache_dir:
            self.download_all()

    def download_all(self):
        print(f"Downloading {len(self.dataset)} items to {self.cache_dir}...")
        from tqdm.auto import tqdm
        from concurrent.futures import ThreadPoolExecutor
        
        def process_item(item):
            # Download all audio columns
            for key in ["hartf_front", "hartf_back", "hrtf_clean"]:
                source = item[key]
                if isinstance(source, str):
                    filename = source.replace("/", "_").replace(":", "_")
                    dest_path = os.path.join(self.cache_dir, filename)
                    if not os.path.exists(dest_path):
                        if source.startswith("http"):
                            url = source
                        else:
                            url = f"{self.base_url}/{source}"
                        try:
                            self._download_with_retry(url, dest_path)
                        except Exception as e:
                            print(f"Failed to download {url}: {e}")

        # Use threading for faster downloads
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(tqdm(executor.map(process_item, self.dataset), total=len(self.dataset)))
        print("Download complete.")

    def __len__(self):
        return len(self.dataset)

    def _process_wav(self, wav, sr):
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
            
        if sr != self.target_sr:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            wav = self.resampler(wav)
        return wav

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
            
            # Check local cache
            cache_path = None
            if self.cache_dir:
                # Create a safe filename from URL
                filename = source.replace("/", "_").replace(":", "_")
                cache_path = os.path.join(self.cache_dir, filename)
                
                if os.path.exists(cache_path):
                    try:
                        wav, sr = torchaudio.load(cache_path)
                        return self._process_wav(wav, sr)
                    except Exception as e:
                        print(f"Error loading from cache {cache_path}: {e}. Re-downloading.")
            
            try:
                # Attempt direct load (if no cache or cache failed)
                # If we have a cache_dir, we prefer downloading to it.
                if self.cache_dir:
                    self._download_with_retry(url, cache_path)
                    wav, sr = torchaudio.load(cache_path)
                else:
                    # Direct load from URL (no cache)
                    # Note: torchaudio.load might fail with 429 if it uses HTTP internally without retries.
                    # If it fails, we fall back to download with retry to a temp file.
                    wav, sr = torchaudio.load(url)
                    
            except Exception as e:
                # Fallback: download to tmp or cache with retry
                if self.cache_dir:
                     # We already tried downloading to cache above if cache_dir was set.
                     # If we are here, it means _download_with_retry failed or torchaudio.load(cache_path) failed.
                     # If _download_with_retry failed, it raised exception.
                     # So we probably won't reach here if cache_dir is set, unless direct load was attempted (logic above).
                     # Let's refine the logic.
                     pass
                
                # If no cache dir, use temp file
                import tempfile
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                try:
                    self._download_with_retry(url, tmp_path)
                    wav, sr = torchaudio.load(tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
        elif isinstance(source, dict) and "array" in source:
             # If we fell back to standard load_dataset, it returns dict
             wav = torch.from_numpy(source["array"]).float()
             sr = source["sampling_rate"]
        else:
            raise ValueError(f"Unknown audio source: {source}")

        return self._process_wav(wav, sr)

    def _download_with_retry(self, url, dest_path):
        import requests
        import time
        
        max_retries = 5
        backoff_factor = 1.0
        
        for i in range(max_retries):
            try:
                with requests.get(url, stream=True) as r:
                    if r.status_code == 429:
                        raise requests.exceptions.RequestException("Too Many Requests")
                    r.raise_for_status()
                    with open(dest_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): 
                            f.write(chunk)
                return # Success
            except Exception as e:
                if i == max_retries - 1:
                    print(f"Download failed for {url} after {max_retries} attempts: {e}")
                    raise e
                
                wait_time = backoff_factor * (2 ** i)
                # print(f"Download failed ({e}), retrying in {wait_time}s...")
                time.sleep(wait_time)

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
    
    cache_dir = "data/audio_cache"
    
    train_ds = BinauralDataset(
        dataset_name=config.data.dataset_name,
        split="train",
        target_sample_rate=config.audio.sample_rate,
        max_items=config.data.max_items,
        cache_dir=cache_dir,
        download=config.data.download
    )
    
    val_ds = BinauralDataset(
        dataset_name=config.data.dataset_name,
        split="validation", # "validation" in metadata.csv
        target_sample_rate=config.audio.sample_rate,
        max_items=config.data.max_items,
        cache_dir=cache_dir,
        download=config.data.download
    )
    
    return train_ds, val_ds
