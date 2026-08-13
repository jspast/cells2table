from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from cells2table.utils.inference import InferenceRuntime, enabled_inference_runtimes

try:
    from transformers import PreTrainedModel

    HAS_TRANSFORMERS = True
    enabled_inference_runtimes.add(InferenceRuntime.TRANSFORMERS)

except ImportError:
    HAS_TRANSFORMERS = False


from cells2table.models.tasks.base import BaseModel
from cells2table.utils.download import DownloadOption, select_download_option


class TransformersModel(BaseModel, ABC):
    """Base interface for Transformers models."""

    _transformers_path: ClassVar[str]
    _transformers_download_options: ClassVar[list[DownloadOption]]

    _transformers_model: PreTrainedModel

    def _transformers_init(self, model_path: Path | str | None = None) -> None:
        self.model_path = self._transformers_download() if model_path is None else Path(model_path)

    @classmethod
    def _transformers_download(cls, *, local_dir: Path | str | None = None) -> Path:
        return select_download_option(cls._transformers_download_options).download(
            local_dir=local_dir
        )

    @classmethod
    def supported_inference_runtimes(cls) -> list[InferenceRuntime]:
        runtimes = super().supported_inference_runtimes()
        runtimes.insert(0, InferenceRuntime.TRANSFORMERS)
        return runtimes

    @abstractmethod
    def _transformers_run(self, input: Any):
        pass
