from abc import ABC, abstractmethod
from typing import Any, ClassVar, NamedTuple

import numpy as np

from cells2table.models.tasks.base import BaseModel


class ClassificationResult(NamedTuple):
    """Result type for classification models."""

    cls: str
    confidence: np.float32


class ClassificationModel(BaseModel, ABC):
    """Base interface for table classification models."""

    classes: ClassVar[list[str]]

    @abstractmethod
    def __call__(self, input: Any) -> list[ClassificationResult]:
        pass
