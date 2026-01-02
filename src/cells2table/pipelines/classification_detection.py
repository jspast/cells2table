import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional

from ..datamodels import Table
from ..models.tasks import ClassificationModel, DetectionModel
from .base import BasePipeline

logger = logging.getLogger(__name__)


class ClassificationDetectionPipeline(BasePipeline, ABC):
    """Base class for standard classification and detection pipelines."""

    classification_model: ClassificationModel
    detection_models: list[DetectionModel]

    @abstractmethod
    def __init__(self, models_path: Optional[Path | str] = None) -> None:
        """Initialize the models."""

        pass

    def __call__(self, input: Iterable[Any]) -> list[Table]:
        """Run the pipeline."""

        cls_images = [[] for c in self.classification_model.classes]
        cls_detections = [[] for c in self.classification_model.classes]
        cls_current_idx = [0 for c in self.classification_model.classes]
        output = []

        cls_result = self.classification_model(input)

        # Run the classification model for each image
        for i, (img, p) in enumerate(zip(input, cls_result)):
            cls_images[self.assigned_model_idx(p.cls, self.detection_models)].append(img)
            logger.info("Image %d classified as %s with %.4f confidence", i, p.cls, p.confidence)

        # Run the detection model for each image
        for i in range(len(self.classification_model.classes)):
            if len(cls_images[i]):
                cls_detections[i] = self.detection_models[i](cls_images[i])

        # Combine results
        for i in range(len(cls_result)):
            model_idx = self.assigned_model_idx(cls_result[i].cls, self.detection_models)
            cells_det = cls_detections[model_idx][cls_current_idx[model_idx]]
            cls_current_idx[model_idx] += 1
            output.append(Table.from_detections(cells_det))

        return output

    @staticmethod
    def assigned_model_idx(pred_cls: str, models: list[DetectionModel]) -> int:
        """Return the index of the first model appropriate for the class."""

        for idx, model in enumerate(models):
            if pred_cls in model.classes:
                return idx

        raise ValueError(f"No model can be assigned for class {pred_cls}")
