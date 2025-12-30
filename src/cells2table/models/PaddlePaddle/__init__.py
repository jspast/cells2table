from .cell_detection import (
    PaddlePaddleCellDetection,
    PaddlePaddleWiredCellDetection,
    PaddlePaddleWirelessCellDetection,
)
from .pipeline import PaddlePaddleTablePipeline
from .table_classification import PaddlePaddleTableClassification

__all__ = [
    PaddlePaddleCellDetection,
    PaddlePaddleWiredCellDetection,
    PaddlePaddleWirelessCellDetection,
    PaddlePaddleTablePipeline,
    PaddlePaddleTableClassification,
]
