import argparse
from huggingface_hub import HfApi, create_repo
import os

def upload_model(model_path, config_path, repo_id, token=None, commit_message="Upload model"):
    api = HfApi(token=token)
    
    # Create repo if it doesn't exist
    print(f"Checking/Creating repo {repo_id}...")
    try:
        create_repo(repo_id, repo_type="model", token=token, exist_ok=True)
        print(f"Repo {repo_id} is ready.")
    except Exception as e:
        print(f"Error creating/checking repo: {e}")
        return

    # Upload Model
    if os.path.exists(model_path):
        print(f"Uploading model from {model_path}...")
        try:
            api.upload_file(
                path_or_fileobj=model_path,
                path_in_repo="best_model.pt",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message
            )
            print("Model uploaded successfully.")
        except Exception as e:
            print(f"Error uploading model: {e}")
    else:
        print(f"Model file not found at {model_path}")

    # Upload Config
    if config_path and os.path.exists(config_path):
        print(f"Uploading config from {config_path}...")
        try:
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="config.yaml",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message
            )
            print("Config uploaded successfully.")
        except Exception as e:
            print(f"Error uploading config: {e}")
    else:
        print(f"Config file not found at {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, default="results/best_model.pt", help="Path to the model file")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face Repo ID (e.g., username/model-name)")
    parser.add_argument("--token", type=str, help="Hugging Face Write Token (optional if logged in via CLI)")
    parser.add_argument("--commit_message", type=str, default="Upload trained model", help="Commit message")
    
    args = parser.parse_args()
    
    upload_model(args.model_path, args.config_path, args.repo_id, args.token, args.commit_message)
