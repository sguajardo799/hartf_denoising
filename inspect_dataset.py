import os
from datasets import load_dataset

def inspect_dataset():
    dataset_name = "sguajardo799/BinauralDenoising2025"
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Load the dataset (streaming=True to avoid downloading everything just for inspection if it's huge)
        # However, for structure inspection, loading the first split is usually fine.
        # If it requires authentication, the user environment should have it or I might fail.
        # Assuming public or auth is handled.
        ds = load_dataset(dataset_name, split="train", streaming=True)
        
        print("\nDataset Features:")
        print(ds.features)
        
        print("\nFirst Example:")
        example = next(iter(ds))
        for key, value in example.items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    Shape: {value.shape}")
            elif isinstance(value, dict) and 'array' in value:
                # Likely an Audio feature
                print(f"    Audio Array Shape: {value['array'].shape}")
                print(f"    Sampling Rate: {value['sampling_rate']}")
            else:
                print(f"    Value (preview): {str(value)[:100]}")

    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    inspect_dataset()
