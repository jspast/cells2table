from pathlib import Path

from cells2table.models.PaddlePaddle import (
    PaddlePaddleTableClassificationModel,
    PaddlePaddleWiredCellDetectionModel,
    PaddlePaddleWirelessCellDetectionModel,
)
from cells2table.pipelines.classification_detection import ClassificationDetectionPipeline
from cells2table.utils.download import combine_download_options, select_download_option
from cells2table.utils.inference import InferenceRuntime, select_inference_runtime


class PaddlePaddleTablePipeline(ClassificationDetectionPipeline):
    """A table pipeline combining PaddlePaddle classification and detection models."""

    _default_runtime = InferenceRuntime.OPENCV

    _onnx_dirname = "jspast--paddlepaddle-table-models-onnx"

    def __init__(
        self,
        models_path: Path | str | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> None:
        """Initialize models from the provided path or download them."""

        runtime = (
            select_inference_runtime(
                PaddlePaddleTableClassificationModel.supported_inference_runtimes()
            )
            if runtime is None
            else runtime
        )

        models_path = self.download(runtime) if models_path is None else Path(models_path)

        self.classification_model = PaddlePaddleTableClassificationModel(models_path, runtime)
        self.detection_models = [
            PaddlePaddleWiredCellDetectionModel(models_path, runtime),
            PaddlePaddleWirelessCellDetectionModel(models_path, runtime),
        ]

    @classmethod
    def download(
        cls,
        runtime: InferenceRuntime,
        local_dir: Path | str | None = None,
    ) -> Path:
        match runtime:
            case InferenceRuntime.ONNXRUNTIME | InferenceRuntime.OPENCV:
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
