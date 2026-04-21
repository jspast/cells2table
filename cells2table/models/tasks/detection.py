from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, NamedTuple

import numpy as np

from cells2table.models.tasks.base import BaseModel


class DetectionResult(NamedTuple):
    """Result type for a detection with no class."""

    bbox: np.ndarray
    confidence: float


def filter_detections(
    detections: Iterable[DetectionResult],
    conf_threshold: float,
) -> list[DetectionResult]:
    return [d for d in detections if d.confidence > conf_threshold]


class DetectionModel(BaseModel, ABC):
    """Base interface for detection models."""

    classes: list[str] = []

    @abstractmethod
    def __call__(
        self,
        input: Any,
        conf_threshold: float = 0.5,
    ) -> list[Iterator[DetectionResult]]:
        pass
