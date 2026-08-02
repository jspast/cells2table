import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import ClassVar

import cv2
import numpy as np
from numpy.typing import NDArray

from cells2table.models.runtimes.onnx import OnnxModel
from cells2table.models.runtimes.opencv import OpencvModel
from cells2table.models.tasks import DetectionModel, DetectionResult
from cells2table.utils.download import DownloadOption, DownloadPlatform
from cells2table.utils.inference import InferenceRuntime

HF_REPO_ID = "jspast/paddlepaddle-table-models-onnx"

logger = logging.getLogger(__name__)


class PaddlePaddleCellDetectionModel(DetectionModel, OnnxModel, OpencvModel):
    """Table cell detection model from PaddlePaddle."""

    _max_detections = 300
    _input_shape = (640, 640)

    _onnx_input_names = ("im_shape", "image", "scale_factor")
    _onnx_output_names = ("fetch_name_0", "fetch_name_1")

    _default_runtime = InferenceRuntime.OPENCV

    def __init__(
        self,
        runtime: InferenceRuntime = _default_runtime,
        model_path: Path | str | None = None,
    ) -> None:
        match runtime:
            case InferenceRuntime.ONNX:
                self._onnx_init(model_path)
                self._run_fn = self._onnx_run
            case InferenceRuntime.OPENCV:
                self._opencv_init(model_path)
                self._run_fn = self._opencv_run

    def __call__(
        self,
        input: Sequence[NDArray[np.uint8]],
        conf_threshold: float = 0.5,
    ) -> list[Iterator[DetectionResult]]:
        return self._run_fn(input)

    def _onnx_run(
        self,
        input: Sequence[NDArray[np.uint8]],
        conf_threshold: float = 0.5,
    ) -> list[Iterator[DetectionResult]]:
        logger.debug("Started preprocessing")

        original_shapes = []
        scale_factors = []
        for img in input:
            original_shape = img.shape[:2]
            original_shapes.append(original_shape)
            scale_factors.append(tuple(original_shape[i] / self._input_shape[i] for i in range(2)))

        imgs = self.preprocess(input)

        input_dict = dict(zip(self._onnx_input_names, [original_shapes, imgs, scale_factors]))

        logger.debug("Done preprocessing")
        logger.debug("Started running the model")

        output = self._onnx_session.run(self._onnx_output_names, input_dict)

        logger.debug("Done running the model")
        logger.debug("Started postprocessing")

        result = self.postprocess(output, scale_factors, conf_threshold)

        logger.debug("Done postprocessing")

        return result

    def _opencv_run(
        self,
        input: Sequence[NDArray[np.uint8]],
        conf_threshold: float = 0.5,
    ) -> list[Iterator[DetectionResult]]:
        logger.debug("Started preprocessing")

        original_shapes = []
        scale_factors = []
        for img in input:
            original_shape = img.shape[:2]
            original_shapes.append(original_shape)
            scale_factor = np.array(
                tuple(original_shape[i] / self._input_shape[i] for i in range(2))
            )
            scale_factors.append(scale_factor)

        imgs = self.preprocess(input)
        self._opencv_net.setInput(np.array(original_shapes, dtype=np.float32), name="im_shape")
        self._opencv_net.setInput(imgs, name="image")
        self._opencv_net.setInput(np.array(scale_factors), name="scale_factor")

        logger.debug("Done preprocessing")
        logger.debug("Started running the model")

        output = self._opencv_net.forward()

        logger.debug("Done running the model")
        logger.debug("Started postprocessing")

        result = self.postprocess(
            [output, np.repeat(self._max_detections, len(input))],
            scale_factors,
            conf_threshold,
        )

        logger.debug("Done postprocessing")

        return result

    def preprocess(self, input: Sequence[NDArray[np.uint8]]) -> NDArray:
        params = cv2.dnn.Image2BlobParams(
            scalefactor=1.0 / 255.0,
            size=(640, 640),
            swapRB=False,
            ddepth=cv2.CV_32F,
            datalayout=cv2.DNN_LAYOUT_NCHW,
            mode=cv2.dnn.DNN_PMODE_NULL,
        )

        return cv2.dnn.blobFromImagesWithParams(input, params)

    @classmethod
    def postprocess(
        cls,
        pred: Sequence,
        scale_factors: Sequence[tuple[int, int] | NDArray],
        conf_threshold: float,
    ) -> list[Iterator[DetectionResult]]:
        last_cell_idx = 0
        cells = pred[0]

        generators = []

        for i, count in enumerate(pred[1]):
            c = cells[last_cell_idx : last_cell_idx + count]
            c = c[c[:, 1] > conf_threshold]
            last_cell_idx += count

            if not c.size:
                generators.append(iter([]))
                continue

            sx, sy = scale_factors[i]
            scores = c[:, 1]
            boxes = c[:, 2:]
            boxes[:, [0, 2]] *= sy
            boxes[:, [1, 3]] *= sx

            generators.append((DetectionResult(box, score) for box, score in zip(boxes, scores)))

        return generators


class PaddlePaddleWiredCellDetectionModel(PaddlePaddleCellDetectionModel):
    classes: ClassVar[list[str]] = ["wired"]

    _onnx_path: ClassVar[str] = "wired_table_cell_det.onnx"
    _onnx_download_options: ClassVar[list[DownloadOption]] = [
        DownloadOption(DownloadPlatform.HUGGINGFACE, HF_REPO_ID, (_onnx_path,)),
    ]


class PaddlePaddleWirelessCellDetectionModel(PaddlePaddleCellDetectionModel):
    classes: ClassVar[list[str]] = ["wireless"]

    _onnx_path: ClassVar[str] = "wireless_table_cell_det.onnx"
    _onnx_download_options: ClassVar[list[DownloadOption]] = [
        DownloadOption(DownloadPlatform.HUGGINGFACE, HF_REPO_ID, (_onnx_path,)),
    ]
