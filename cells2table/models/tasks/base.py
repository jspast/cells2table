from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    """Base interface for models of any type."""

    @abstractmethod
    def __init__(self, *, model_path: Path | str | None = None) -> None:
        pass

    @abstractmethod
    def __call__(self, input: Any):
        pass
