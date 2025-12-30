from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from ...utils.table import BoundingBox, Cell, Table
from ..utils.tasks import ClassificationResult, DetectionResult
from .cell_detection import PaddlePaddleWiredCellDetection, PaddlePaddleWirelessCellDetection
from .table_classification import PaddlePaddleTableClassification


class PaddlePaddleTablePipeline:
    def __init__(self):
        self.cls_predictor = PaddlePaddleTableClassification()
        self.wired_predictor = PaddlePaddleWiredCellDetection()
        self.wireless_predictor = PaddlePaddleWirelessCellDetection()

    def __call__(self, input: Iterable[NDArray[np.uint8]]) -> list[Table]:
        wired_images, wireless_images, output = [], [], []

        cls_result: list[ClassificationResult] = self.cls_predictor(input)

        for img, p in zip(input, cls_result):
            (wired_images if p.cls == "wired" else wireless_images).append(img)

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

            output.append(self.build_table(cells_det))

        return output

    @staticmethod
    def build_table(cells_det: Iterable[DetectionResult]) -> Table:
        table = Table()

        for cell_det in cells_det:
            bbox = BoundingBox.from_array(cell_det.bbox)
            cell = Cell(bbox=bbox, row=0, col=0)
            table.cells.append(cell)

        return table
