from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from cells2table.models.tasks.base import BaseModel, Prediction


@dataclass(frozen=True, slots=True)
class Classification(Prediction):
    """Result type for classification models.

    Attributes:
        confidence: Confidence score (0-1).
        id: Class ID.
    """

    id: int


class ClassificationModel(BaseModel, ABC):
    """Base interface for table classification models.

    Attributes:
        id2label: Mapping from class ID to human-readable label.
    """

    id2label: ClassVar[dict[int, str]]

    @abstractmethod
    def __call__(self, input: Any) -> list[Classification]:
        """Run classification inference.

        Args:
            input: List of images.

        Returns:
            List of Classification results, one per image.
        """
