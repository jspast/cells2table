from pathlib import Path
from typing import Optional

from cells2table.models.PaddlePaddle import (
    PaddlePaddleTableClassificationModel,
    PaddlePaddleWiredCellDetectionModel,
    PaddlePaddleWirelessCellDetectionModel,
)
from cells2table.pipelines.classification_detection import ClassificationDetectionPipeline


class PaddlePaddleTablePipeline(ClassificationDetectionPipeline):
    """A table pipeline combining PaddlePaddle classification and detection models."""

    def __init__(self, models_path: Optional[Path | str] = None) -> None:
        """Initialize models from the provided path or download them."""

        models_path = self.download() if models_path is None else Path(models_path)

        self.classification_model = PaddlePaddleTableClassificationModel(models_path)
        self.detection_models = [
            PaddlePaddleWiredCellDetectionModel(models_path),
            PaddlePaddleWirelessCellDetectionModel(models_path),
        ]

    @classmethod
    def download(cls, *, local_dir: Path | str | None = None) -> Path:
        PaddlePaddleTableClassificationModel.download(local_dir=local_dir)
        PaddlePaddleWiredCellDetectionModel.download(local_dir=local_dir)
        path = PaddlePaddleWirelessCellDetectionModel.download(local_dir=local_dir)
        return path
