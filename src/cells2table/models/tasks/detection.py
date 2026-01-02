from abc import ABC, abstractmethod
from typing import Any, Iterator, NamedTuple

import numpy as np

from .base import BaseModel


class DetectionResult(NamedTuple):
    """Result type for a detection with no class."""

    bbox: np.ndarray
    confidence: float


class DetectionModel(BaseModel, ABC):
    """Base interface for detection models."""

    classes: list[str] = []

    @abstractmethod
    def __call__(self, input: Any) -> list[Iterator[DetectionResult]]:
        pass
