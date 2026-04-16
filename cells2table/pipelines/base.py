from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BasePipeline(ABC):
    """Base interface for pipelines of any type."""

    @abstractmethod
    def __init__(self, models_path: Path | str | None = None) -> None:
        pass

    @abstractmethod
    def __call__(self, input: Any):
        pass

    @classmethod
    @abstractmethod
    def download(cls, *, local_dir: Path | str | None = None) -> Path:
        pass
