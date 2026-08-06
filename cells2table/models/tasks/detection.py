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


def filter_detections(
    detections: Iterable[Detection],
    conf_threshold: float,
) -> list[Detection]:
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
