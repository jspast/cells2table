from pathlib import Path

from cells2table.models.PaddlePaddle import (
    PaddlePaddleTableClassificationModel,
    PaddlePaddleWiredCellDetectionModel,
    PaddlePaddleWirelessCellDetectionModel,
)
from cells2table.pipelines.classification_detection import ClassificationDetectionPipeline
from cells2table.utils.inference import InferenceRuntime


class PaddlePaddleTablePipeline(ClassificationDetectionPipeline):
    """A table pipeline combining PaddlePaddle classification and detection models."""

    _dirname = "jspast--paddlepaddle-table-models-onnx"

    def __init__(
        self,
        models_path: Path | str | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> None:
        """Initialize models from the provided path or download them."""

        runtime = (
            PaddlePaddleTableClassificationModel._default_runtime if runtime is None else runtime
        )

        download_options_attr: str
        match runtime:
            case InferenceRuntime.ONNX | InferenceRuntime.OPENCV:
                download_options_attr = "_onnx_download_options"

        models_path = (
            self.download(download_options_attr) if models_path is None else Path(models_path)
        )

        self.classification_model = PaddlePaddleTableClassificationModel(runtime, models_path)
        self.detection_models = [
            PaddlePaddleWiredCellDetectionModel(runtime, models_path),
            PaddlePaddleWirelessCellDetectionModel(runtime, models_path),
        ]

    @classmethod
    def download(
        cls,
        download_options_attr: str = "_onnx_download_options",
        local_dir: Path | str | None = None,
    ) -> Path:
        pipeline_dir = None if local_dir is None else Path(local_dir) / cls._dirname

        getattr(PaddlePaddleTableClassificationModel, download_options_attr)[0].download(
            local_dir=pipeline_dir
        )
        getattr(PaddlePaddleWiredCellDetectionModel, download_options_attr)[0].download(
            local_dir=pipeline_dir
        )
        path = getattr(PaddlePaddleWirelessCellDetectionModel, download_options_attr)[0].download(
            local_dir=pipeline_dir
        )
        return path
