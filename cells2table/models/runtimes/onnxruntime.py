from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # ty:ignore[invalid-assignment]

from cells2table.models.tasks.base import BaseModel
from cells2table.utils.download import DownloadOption, select_download_option


class ONNXRuntimeModel(BaseModel, ABC):
    """Base interface for ONNX Runtime models."""

    _onnx_path: ClassVar[str]
    _onnx_download_options: ClassVar[list[DownloadOption]]

    _onnxruntime_session: ort.InferenceSession

    def _onnxruntime_init(self, model_path: Path | str | None = None) -> None:
        self.model_path = self._onnx_download() if model_path is None else Path(model_path)

        providers_priority = [
            "CUDAExecutionProvider",
            "MIGraphXExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        available_providers = ort.get_available_providers()

        self._onnxruntime_session = ort.InferenceSession(
            self.model_path / self._onnx_path,
            providers=[p for p in providers_priority if p in available_providers],
        )

    @classmethod
    def _onnx_download(cls, *, local_dir: Path | str | None = None) -> Path:
        return select_download_option(cls._onnx_download_options).download(local_dir=local_dir)

    @abstractmethod
    def _onnxruntime_run(self, input: Any):
        pass

    # Useful implementation functions:
    #   self._onnx_session.get_inputs()
    #   self._onnx_session.get_inputs()[x].shape
    #   self._onnx_session.get_outputs()
