from abc import ABC
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from ..tasks.base import BaseModel


class OnnxModel(BaseModel, ABC):
    """Base interface for ONNX models."""

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        if model_path is None:
            model_path = self.download()

        providers_priority = [
            "CUDAExecutionProvider",
            "MIGraphXExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        available_providers = ort.get_available_providers()  # type: ignore

        self.session = ort.InferenceSession(
            model_path,
            providers=[p for p in providers_priority if p in available_providers],
        )

        self.scale = 1 / 255.0
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def input_shape(self):
        return self.session.get_inputs()[0].shape[2:]  # assuming NCHW

    @property
    def input_names(self):
        return [v.name for v in self.session.get_inputs()]

    @property
    def output_names(self):
        return [v.name for v in self.session.get_outputs()]

    def preprocess(self, input: Iterable[NDArray[np.uint8]]) -> list[NDArray[np.uint8]]:
        output = []

        for img in input:
            img = cv2.resize(img, dsize=self.input_shape, interpolation=cv2.INTER_LANCZOS4)
            img = (img.astype(np.float32) * self.scale - self.mean) / self.std  # Normalize
            img = img.transpose(2, 0, 1)  # HWC to CHW
            output.append(img)

        return output
