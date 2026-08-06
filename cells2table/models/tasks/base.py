from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Base interface for models of any type."""

    @abstractmethod
    def __init__(self, *, model_path: Path | str | None = None) -> None:
        pass

    @abstractmethod
    def __call__(self, input: Any):
        pass


@dataclass(frozen=True, slots=True)
class Prediction:
    """Base result type for model predictions."""

    confidence: np.float32
