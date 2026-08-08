from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from cells2table.models.tasks.base import BaseModel, Prediction


@dataclass(frozen=True, slots=True)
class Detection(Prediction):
    """Result type for a detection with no class."""

    bbox: np.ndarray


@dataclass(frozen=True, slots=True)
class ClassifiedDetection(Detection):
    """Result type for a detection with a class."""

    id: int


def filter_detections(
    detections: Iterable[Detection | ClassifiedDetection],
    conf_threshold: float,
) -> list[Detection | ClassifiedDetection]:
    return [d for d in detections if d.confidence > conf_threshold]


class DetectionModel(BaseModel, ABC):
    """Base interface for detection models."""

    @abstractmethod
    def __call__(
        self,
        input: Any,
        conf_threshold: float = 0.5,
    ) -> list[Iterator[Detection]]:
        pass


class ClassifiedDetectionModel(BaseModel, ABC):
    """Base interface for detection with classes models."""

    @abstractmethod
    def __call__(
        self,
        input: Any,
        conf_threshold: float = 0.5,
    ) -> list[Iterator[ClassifiedDetection]]:
        pass
