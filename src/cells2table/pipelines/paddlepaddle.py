import logging
from pathlib import Path
from typing import Optional

from ..models.PaddlePaddle import (
    PaddlePaddleTableClassificationModel,
    PaddlePaddleWiredCellDetectionModel,
    PaddlePaddleWirelessCellDetectionModel,
)
from .classification_detection import ClassificationDetectionPipeline

logger = logging.getLogger(__name__)


class PaddlePaddleTablePipeline(ClassificationDetectionPipeline):
    """A table pipeline combining PaddlePaddle classification and detection models."""

    def __init__(self, models_path: Optional[Path | str] = None) -> None:
        """Initialize models from the provided path or download them.

        As the models are all in the same repository, do the download only once.
        """

        self.detection_models = []
        cls_path, wired_path, wireless_path = None, None, None

        if models_path is not None:
            models_path = Path(models_path)
            cls_path = (
                models_path / PaddlePaddleTableClassificationModel.download_options.model_path
            )

        self.classification_model = PaddlePaddleTableClassificationModel(cls_path)

        wired_path = (
            self.classification_model.model_path.parent
            / PaddlePaddleWiredCellDetectionModel.download_options.model_path
        )
        wireless_path = (
            self.classification_model.model_path.parent
            / PaddlePaddleWirelessCellDetectionModel.download_options.model_path
        )

        self.detection_models.append(PaddlePaddleWiredCellDetectionModel(wired_path))
        self.detection_models.append(PaddlePaddleWirelessCellDetectionModel(wireless_path))
