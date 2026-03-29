from abc import ABC, abstractmethod
from typing import Any, NamedTuple

from .base import BaseModel


class ClassificationResult(NamedTuple):
    """Result type for classification models."""

    cls: str
    confidence: float


class ClassificationModel(BaseModel, ABC):
    """Base interface for table classification models."""

    classes: list[str]

    @abstractmethod
    def __call__(self, input: Any) -> list[ClassificationResult]:
        pass
