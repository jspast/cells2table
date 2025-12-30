from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.download import download_hf_model
from ..utils.runtimes import OnnxModel
from ..utils.tasks import ClassificationModel, ClassificationResult

HF_REPO_ID = "jspast/paddlepaddle-table-models-onnx"


class PaddlePaddleTableClassification(ClassificationModel, OnnxModel):
    @staticmethod
    def download() -> Path:
        return download_hf_model(HF_REPO_ID) / "table_cls.onnx"

    def __call__(self, input: Iterable[NDArray[np.uint8]]) -> list[ClassificationResult]:
        input = self.preprocess(input)

        input_dict = dict(zip(self.input_names, [input]))

        output = self.session.run(self.output_names, input_dict)[0]

        return self.postprocess(output)  # type: ignore

    @staticmethod
    def postprocess(pred: Sequence[Sequence[float]]) -> list[ClassificationResult]:
        return [
            ClassificationResult({0: "wired", 1: "wireless"}[np.argmax(p)], max(p)) for p in pred
        ]
