from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cells2table.utils.download import DownloadOptions


class BaseModel(ABC):
    """Base interface for models of any type."""

    model_path: Path

    @abstractmethod
    def __init__(self, model_path: Path | str | None = None) -> None:
        pass

    @abstractmethod
    def __call__(self, input: Any):
        pass

    @classmethod
    @abstractmethod
    def get_download_options(cls) -> DownloadOptions:
        pass

    @classmethod
    def download(cls, *, local_dir: Path | str | None = None) -> Path:
        return cls.get_download_options().download(local_dir=local_dir)
