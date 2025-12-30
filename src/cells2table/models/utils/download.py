from pathlib import Path

from huggingface_hub import snapshot_download


def download_hf_model(repo_id: str) -> Path:
    download_path = snapshot_download(repo_id=repo_id)

    return Path(download_path)
