from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from cells2table.models.tasks.base import BaseModel, Prediction


@dataclass(frozen=True, slots=True)
class Detection(Prediction):
    """Result type for object detection.

    Attributes:
        confidence: Confidence score (0-1).
        bbox: Bounding box as [x_min, y_min, x_max, y_max] array.
    """

    bbox: np.ndarray


@dataclass(frozen=True, slots=True)
class ClassifiedDetection(Detection):
    """Result type for object detection with classification.

    Attributes:
        confidence: Confidence score (0-1).
        bbox: Bounding box as [x_min, y_min, x_max, y_max] array.
        id: Class ID.
    """

    id: int


def filter_detections(
    detections: Iterable[Detection | ClassifiedDetection],
    conf_threshold: float,
) -> list[Detection | ClassifiedDetection]:
    """Filter detections by confidence threshold.

    Args:
        detections: List of detections.
        conf_threshold: Minimum confidence score (0-1).

    Returns:
        Filtered detections above threshold.
    """
    return [d for d in detections if d.confidence > conf_threshold]


class DetectionModel(BaseModel, ABC):
    """Base interface for object detection models."""

    @abstractmethod
    def __call__(
        self,
        input: Any,
        conf_threshold: float = 0.5,
    ) -> list[Iterator[Detection]]:
        """Run detection inference.

        Args:
            input: List of images.
            conf_threshold: Minimum confidence threshold (0-1).

        Returns:
            List of iterators, one per image, yielding Detection objects.
        """


class ClassifiedDetectionModel(BaseModel, ABC):
    """Base interface for detection models with class labels."""

    @abstractmethod
    def __call__(
        self,
        input: Any,
        conf_threshold: float = 0.5,
    ) -> list[Iterator[ClassifiedDetection]]:
        """Run detection inference with classification.

        Args:
            input: List of images.
            conf_threshold: Minimum confidence threshold (0-1).

        Returns:
            List of iterators, one per image, yielding ClassifiedDetection objects.
        """
