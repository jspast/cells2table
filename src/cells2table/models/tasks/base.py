from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from ..download import DownloadOptions, download


class BaseModel(ABC):
    """Base interface for models of any type."""

    model_path: Path
    download_options: Optional[DownloadOptions] = None

    @abstractmethod
    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        pass

    @abstractmethod
    def __call__(self, input: Any):
        pass

    @classmethod
    def download(cls) -> Path:
        if cls.download_options is not None:
            return download(cls.download_options)
        else:
            raise NotImplementedError(
                "Download is not implemented for this model. Please provide a path."
            )
