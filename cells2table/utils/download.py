import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class DownloadPlatform(StrEnum):
    HUGGINGFACE = "huggingface"


enabled_download_platforms: list[DownloadPlatform] = [DownloadPlatform.HUGGINGFACE]


@dataclass(frozen=True)
class DownloadOption:
    platform: DownloadPlatform
    repo_id: str
    files: tuple[str, ...] | None = None

    def download(self, *, local_dir: Path | str | None = None) -> Path:
        match self.platform:
            case DownloadPlatform.HUGGINGFACE:
                path = hf_download(
                    self.repo_id,
                    files=None if self.files is None else list(self.files),
                    local_dir=local_dir,
                )

        return path


def combine_download_options(options: list[DownloadOption]) -> DownloadOption:
    platform = options[0].platform
    repo_id = options[0].repo_id
    files = set(() if options[0].files is None else options[0].files)

    for o in options[1:]:
        if o.platform == platform and o.repo_id == repo_id and o.files is not None:
            files.update(o.files)

    return DownloadOption(platform, repo_id, tuple(files))


def select_download_option(supported: list[DownloadOption]) -> DownloadOption:
    for p in enabled_download_platforms:
        for o in supported:
            if p == o.platform:
                return o

    raise (ValueError("No supported download option found. Check enabled_download_platforms."))


def hf_download(
    repo_id: str,
    *,
    files: list[str] | None = None,
    local_dir: Path | str | None = None,
) -> Path:
    """Download a repository from Hugging Face and return its path."""

    try:
        from huggingface_hub import snapshot_download, try_to_load_from_cache
        from huggingface_hub.utils import disable_progress_bars
    except ImportError:
        raise ImportError("huggingface_hub is not installed. Unable to download the model.")

    disable_progress_bars()

    download_path = try_to_load_from_cache(repo_id=repo_id, filename=files[0] if files else "")

    if isinstance(download_path, str):
        logger.info("Repo %s cached, no need to redownload", repo_id)
        download_path = Path(download_path).parent
    else:
        logger.info("Downloading HF repo %s", repo_id)
        download_path = Path(
            snapshot_download(repo_id=repo_id, allow_patterns=files, local_dir=local_dir)
        )

    return Path(download_path)
