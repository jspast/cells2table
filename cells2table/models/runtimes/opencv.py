from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import cv2

from cells2table.models.tasks.base import BaseModel
from cells2table.utils.download import DownloadOption, select_download_option
from cells2table.utils.inference import InferenceRuntime


class OpenCVModel(BaseModel, ABC):
    """Base interface for OpenCV models."""

    _onnx_path: ClassVar[str]
    _onnx_download_options: ClassVar[list[DownloadOption]]

    _opencv_net: cv2.dnn.Net

    def _opencv_init(self, model_path: Path | str | None = None) -> None:
        self.model_path = self._onnx_download() if model_path is None else Path(model_path)

        self._opencv_net = cv2.dnn.readNet(self.model_path / self._onnx_path)

    @classmethod
    def _onnx_download(cls, *, local_dir: Path | str | None = None) -> Path:
        return select_download_option(cls._onnx_download_options).download(local_dir=local_dir)

    @classmethod
    def supported_inference_runtimes(cls) -> list[InferenceRuntime]:
        runtimes = super().supported_inference_runtimes()
        runtimes.insert(0, InferenceRuntime.OPENCV)
        return runtimes

    @abstractmethod
    def _opencv_run(self, input: Any):
        pass
