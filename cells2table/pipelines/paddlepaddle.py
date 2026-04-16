from pathlib import Path

from cells2table.models.PaddlePaddle import (
    PaddlePaddleTableClassificationModel,
    PaddlePaddleWiredCellDetectionModel,
    PaddlePaddleWirelessCellDetectionModel,
)
from cells2table.pipelines.classification_detection import ClassificationDetectionPipeline


class PaddlePaddleTablePipeline(ClassificationDetectionPipeline):
    """A table pipeline combining PaddlePaddle classification and detection models."""

    _dirname = "jspast--paddlepaddle-table-models-onnx"

    def __init__(self, models_path: Path | str | None = None) -> None:
        """Initialize models from the provided path or download them."""

        models_path = self.download() if models_path is None else Path(models_path)

        self.classification_model = PaddlePaddleTableClassificationModel(models_path)
        self.detection_models = [
            PaddlePaddleWiredCellDetectionModel(models_path),
            PaddlePaddleWirelessCellDetectionModel(models_path),
        ]

    @classmethod
    def download(cls, *, local_dir: Path | str | None = None) -> Path:
        pipeline_dir = None if local_dir is None else Path(local_dir) / cls._dirname
        PaddlePaddleTableClassificationModel.download(local_dir=pipeline_dir)
        PaddlePaddleWiredCellDetectionModel.download(local_dir=pipeline_dir)
        path = PaddlePaddleWirelessCellDetectionModel.download(local_dir=pipeline_dir)
        return path
