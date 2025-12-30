from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class BaseModel(ABC):
    """Base interface for models of any type"""

    @abstractmethod
    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        pass

    @staticmethod
    @abstractmethod
    def download() -> Path:
        pass

    @abstractmethod
    def __call__(self, input: Any):
        pass
