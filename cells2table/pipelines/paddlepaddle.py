from pathlib import Path

from cells2table.models.PaddlePaddle import (
    PaddlePaddleTableClassificationModel,
    PaddlePaddleWiredCellDetectionModel,
    PaddlePaddleWirelessCellDetectionModel,
)
from cells2table.pipelines.classification_detection import ClassificationDetectionPipeline
from cells2table.utils.download import combine_download_options, select_download_option
from cells2table.utils.inference import InferenceRuntime


class PaddlePaddleTablePipeline(ClassificationDetectionPipeline):
    """A table pipeline combining PaddlePaddle classification and detection models."""

    _default_runtime = PaddlePaddleTableClassificationModel._default_runtime

    _onnx_dirname = "jspast--paddlepaddle-table-models-onnx"

    def __init__(
        self,
        models_path: Path | str | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> None:
        """Initialize models from the provided path or download them."""

        runtime = self._default_runtime if runtime is None else runtime

        models_path = self.download(runtime) if models_path is None else Path(models_path)

        self.classification_model = PaddlePaddleTableClassificationModel(runtime, models_path)
        self.detection_models = [
            PaddlePaddleWiredCellDetectionModel(runtime, models_path),
            PaddlePaddleWirelessCellDetectionModel(runtime, models_path),
        ]

    @classmethod
    def download(
        cls,
        runtime: InferenceRuntime,
        local_dir: Path | str | None = None,
    ) -> Path:
        match runtime:
            case InferenceRuntime.ONNX | InferenceRuntime.OPENCV:
                pipeline_dir = None if local_dir is None else Path(local_dir) / cls._onnx_dirname
                path = combine_download_options(
                    [
                        select_download_option(
                            PaddlePaddleTableClassificationModel._onnx_download_options
                        ),
                        select_download_option(
                            PaddlePaddleWiredCellDetectionModel._onnx_download_options
                        ),
                        select_download_option(
                            PaddlePaddleWirelessCellDetectionModel._onnx_download_options
                        ),
                    ]
                ).download(local_dir=pipeline_dir)

            case InferenceRuntime.TRANSFORMERS:
                path = select_download_option(
                    PaddlePaddleTableClassificationModel._transformers_download_options
                ).download(local_dir=local_dir)
                select_download_option(
                    PaddlePaddleWiredCellDetectionModel._transformers_download_options
                ).download(local_dir=local_dir)
                select_download_option(
                    PaddlePaddleWirelessCellDetectionModel._transformers_download_options
                ).download(local_dir=local_dir)

        return path

    @staticmethod
    def assigned_model_idx(pred_class_id: int) -> int:
        """Return the index of the appropriate model for the class."""

        return pred_class_id
