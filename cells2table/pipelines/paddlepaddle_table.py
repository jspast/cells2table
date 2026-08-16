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
    """Complete table detection pipeline using PaddlePaddle models.

    This pipeline combines table classification (wired vs. wireless) with
    cell detection.

    The wired detection model is used for tables with visible borders,
    while the wireless model is used for tables without visible borders.
    """

    _default_runtime = InferenceRuntime.OPENCV

    _onnx_dirname = "jspast--paddlepaddle-table-models-onnx"

    def __init__(
        self,
        models_path: Path | str | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> None:
        """Initialize pipeline with PaddlePaddle models.

        Args:
            models_path: Path to directory with model weights.
            runtime: Inference runtime (onnxruntime, opencv, or transformers). If None, auto-selected.

        Raises:
            ValueError: If no inference runtime is available.
        """

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
        """Download models from HuggingFace.

        Args:
            runtime: Inference runtime (determines model format).
            local_dir: Custom download directory.

        Returns:
            Path to downloaded model directory.
        """
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
        """Map classification result to detection model.

        Args:
            pred_class_id: Class ID (0=wired, 1=wireless).

        Returns:
            Detection model index (0 for wired, 1 for wireless).
        """

        return pred_class_id
