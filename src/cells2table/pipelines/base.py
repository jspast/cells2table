from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class BasePipeline(ABC):
    """Base interface for pipelines of any type."""

    @abstractmethod
    def __init__(self, models_path: Optional[Path | str] = None) -> None:
        pass

    @abstractmethod
    def __call__(self, input: Any):
        pass
