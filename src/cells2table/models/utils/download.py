import logging
from pathlib import Path



logger = logging.getLogger(__name__)


def download_hf_model(repo_id: str) -> Path:
    """Download a repository from Hugging Face and return its path."""

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars
    except ImportError:
        raise ImportError("huggingface_hub is not installed. Unable to download the model.")

    disable_progress_bars()

    logger.info("Downloading HF repo %s", repo_id)
    download_path = snapshot_download(repo_id=repo_id)

    return Path(download_path)
