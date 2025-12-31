import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from numpy.typing import NDArray

from ...utils.table import Table
from .cell_detection import PaddlePaddleWiredCellDetection, PaddlePaddleWirelessCellDetection
from .table_classification import PaddlePaddleTableClassification

logger = logging.getLogger(__name__)


class PaddlePaddleTablePipeline:
    """A table pipeline combining PaddlePaddle classification and detection models."""

    def __init__(self, model_path: Optional[Path | str] = None):
        self.cls_predictor = PaddlePaddleTableClassification(model_path)
        self.wired_predictor = PaddlePaddleWiredCellDetection(model_path)
        self.wireless_predictor = PaddlePaddleWirelessCellDetection(model_path)

    def __call__(self, input: Iterable[NDArray[np.uint8]]) -> list[Table]:
        wired_images, wireless_images, output = [], [], []

        cls_result = self.cls_predictor(input)

        for i, (img, p) in enumerate(zip(input, cls_result)):
            (wired_images if p.cls == "wired" else wireless_images).append(img)
            logger.info("Image %d classified as %s", i, p.cls)

        if len(wired_images):
            wired_cells = self.wired_predictor(wired_images)

        if len(wireless_images):
            wireless_cells = self.wireless_predictor(wireless_images)

        wired_idx = 0
        wireless_idx = 0

        for i in range(len(cls_result)):
            if cls_result[i].cls == "wired":
                cells_det = wired_cells[wired_idx]
                wired_idx += 1
            else:
                cells_det = wireless_cells[wireless_idx]
                wireless_idx += 1

            output.append(Table.from_detections(cells_det))

        return output
