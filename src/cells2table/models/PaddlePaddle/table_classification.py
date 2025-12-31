import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..utils.download import download_hf_model
from ..utils.runtimes import OnnxModel
from ..utils.tasks import ClassificationModel, ClassificationResult

HF_REPO_ID = "jspast/paddlepaddle-table-models-onnx"

logger = logging.getLogger(__name__)


class PaddlePaddleTableClassification(ClassificationModel, OnnxModel):
    @staticmethod
    def download() -> Path:
        return download_hf_model(HF_REPO_ID) / "table_cls.onnx"

    def __call__(self, input: Iterable[NDArray[np.uint8]]) -> list[ClassificationResult]:
        logger.debug("Started preprocessing")
        input = self.preprocess(input)

        input_dict = dict(zip(self.input_names, [input]))

        logger.debug("Done preprocessing")
        logger.debug("Started running the model")

        output = self.session.run(self.output_names, input_dict)[0]

        logger.debug("Done running the model")
        logger.debug("Started postprocessing")

        result = self.postprocess(output)  # type: ignore

        logger.debug("Done postprocessing")

        return result

    @staticmethod
    def postprocess(pred: Sequence[Sequence[float]]) -> list[ClassificationResult]:
        return [
            ClassificationResult({0: "wired", 1: "wireless"}[np.argmax(p)], max(p)) for p in pred
        ]
