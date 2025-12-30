from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.download import download_hf_model
from ..utils.runtimes import OnnxModel
from ..utils.tasks import DetectionModel, DetectionResult

HF_REPO_ID = "jspast/paddlepaddle-table-models-onnx"
CONFIDENCE_THRESHOLD = 0.5


class PaddlePaddleCellDetection(DetectionModel, OnnxModel):
    @staticmethod
    def download() -> Path:
        return download_hf_model(HF_REPO_ID) / "table_cls.onnx"

    @property
    def input_shape(self):
        return self.session.get_inputs()[1].shape[2:]  # assuming NCHW

    def __call__(self, input: Iterable[NDArray[np.uint8]]) -> list[Iterator[DetectionResult]]:
        original_shapes = []
        scale_factors = []
        for img in input:
            original_shape = img.shape[:2]
            original_shapes.append(original_shape)
            scale_factors.append(
                tuple(original_shape[i] / self.input_shape[i] for i in range(0, 2))
            )

        imgs = self.preprocess(input)

        input_dict = dict(zip(self.input_names, [original_shapes, imgs, scale_factors]))

        output = self.session.run(self.output_names, input_dict)

        return self.postprocess(output, scale_factors)  # type: ignore

    def postprocess(
        self,
        pred: NDArray,
        scale_factors: Sequence[tuple[int, int]],
    ) -> list[Iterator[DetectionResult]]:
        last_cell_idx = 0
        batch_size = len(pred[1])
        generators = []
        cells = pred[0]

        for i in range(batch_size):
            c = cells[last_cell_idx : last_cell_idx + pred[1][i]]
            c = c[c[:, 1] > CONFIDENCE_THRESHOLD]

            last_cell_idx += pred[1][i]

            if c.size:
                sx, sy = scale_factors[i]
                scores = c[:, 0]
                boxes = c[:, 2:]
                boxes[:, [0, 2]] *= sy
                boxes[:, [1, 3]] *= sx

            generators.append((DetectionResult(box, score) for box, score in zip(boxes, scores)))

        return generators


class PaddlePaddleWiredCellDetection(PaddlePaddleCellDetection):
    @staticmethod
    def download() -> Path:
        return download_hf_model(HF_REPO_ID) / "wired_table_cell_det.onnx"


class PaddlePaddleWirelessCellDetection(PaddlePaddleCellDetection):
    @staticmethod
    def download() -> Path:
        return download_hf_model(HF_REPO_ID) / "wireless_table_cell_det.onnx"
