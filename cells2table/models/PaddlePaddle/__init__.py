from .cell_detection import (
    PaddlePaddleCellDetectionModel,
    PaddlePaddleWiredCellDetectionModel,
    PaddlePaddleWirelessCellDetectionModel,
)
from .table_classification import PaddlePaddleTableClassificationModel

__all__ = [
    "PaddlePaddleCellDetectionModel",
    "PaddlePaddleTableClassificationModel",
    "PaddlePaddleWiredCellDetectionModel",
    "PaddlePaddleWirelessCellDetectionModel",
]
