import logging
from enum import Enum
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)


class DownloadPlatform(Enum):
    HUGGINGFACE = "huggingface"


class DownloadOptions(NamedTuple):
    platform: DownloadPlatform
    repo_id: str
    files: list[str] | None = None

    def download(self, *, local_dir: Path | str | None = None) -> Path:
        match self.platform:
            case DownloadPlatform.HUGGINGFACE:
                path = download_hf_model(self.repo_id, files=self.files, local_dir=local_dir)

        return path


def download_hf_model(
    repo_id: str,
    *,
    files: list[str] | None = None,
    local_dir: Path | str | None = None,
) -> Path:
    """Download a repository from Hugging Face and return its path."""

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars
    except ImportError:
        raise ImportError("huggingface_hub is not installed. Unable to download the model.")

    disable_progress_bars()

    logger.info("Downloading HF repo %s", repo_id)
    download_path = snapshot_download(repo_id=repo_id, allow_patterns=files, local_dir=local_dir)

    return Path(download_path)
