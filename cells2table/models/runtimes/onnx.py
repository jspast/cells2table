from abc import ABC, abstractmethod
from pathlib import Path

import onnxruntime as ort

from cells2table.models.tasks.base import BaseModel


class OnnxModel(BaseModel, ABC):
    """Base interface for ONNX models."""

    @classmethod
    @abstractmethod
    def get_onnx_path(self) -> str:
        pass

    def __init__(self, model_path: Path | str | None = None) -> None:
        self.model_path = self.download() if model_path is None else Path(model_path)

        providers_priority = [
            "CUDAExecutionProvider",
            "MIGraphXExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        available_providers = ort.get_available_providers()

        self.session = ort.InferenceSession(
            self.model_path / self.get_onnx_path(),
            providers=[p for p in providers_priority if p in available_providers],
        )

    @property
    def input_shape(self):
        return self.session.get_inputs()[0].shape[2:]  # assuming NCHW

    @property
    def input_names(self):
        return [v.name for v in self.session.get_inputs()]

    @property
    def output_names(self):
        return [v.name for v in self.session.get_outputs()]
