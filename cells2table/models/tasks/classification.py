from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from cells2table.models.tasks.base import BaseModel, Prediction


@dataclass(frozen=True, slots=True)
class Classification(Prediction):
    """Result type for classification models."""

    id: int


class ClassificationModel(BaseModel, ABC):
    """Base interface for table classification models."""

    id2label: ClassVar[dict[int, str]]

    @abstractmethod
    def __call__(self, input: Any) -> list[Classification]:
        pass
