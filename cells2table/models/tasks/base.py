from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cells2table.utils.inference import InferenceRuntime, select_inference_runtime


class BaseModel(ABC):
    """Base interface for models of any type."""

    @property
    def runtime(self) -> InferenceRuntime:
        return self._runtime

    @abstractmethod
    def __init__(
        self,
        model_path: Path | str | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> None:
        self._runtime = (
            select_inference_runtime(self.supported_inference_runtimes())
            if runtime is None
            else runtime
        )

    @abstractmethod
    def __call__(self, input: Any):
        pass

    @classmethod
    @abstractmethod
    def supported_inference_runtimes(cls) -> list[InferenceRuntime]:
        return []


@dataclass(frozen=True, slots=True)
class Prediction:
    """Base result type for model predictions."""

    confidence: np.float32
