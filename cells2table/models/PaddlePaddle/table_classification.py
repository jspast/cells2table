import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar, override

import cv2
import numpy as np
from numpy.typing import NDArray

from cells2table.models.runtimes.onnxruntime import ONNXRuntimeModel
from cells2table.models.runtimes.opencv import OpencvModel
from cells2table.models.runtimes.transformers import TransformersModel
from cells2table.models.tasks import Classification, ClassificationModel
from cells2table.utils.download import DownloadOption, DownloadPlatform
from cells2table.utils.inference import InferenceRuntime

HF_REPO_ID = "jspast/paddlepaddle-table-models-onnx"

logger = logging.getLogger(__name__)


class PaddlePaddleTableClassificationModel(
    ClassificationModel, ONNXRuntimeModel, OpencvModel, TransformersModel
):
    id2label: ClassVar[dict[int, str]] = {0: "wired", 1: "wireless"}

    _input_shape = (224, 224)

    _onnx_path: ClassVar[str] = "table_cls.onnx"
    _onnx_download_options: ClassVar[list[DownloadOption]] = [
        DownloadOption(DownloadPlatform.HUGGINGFACE, HF_REPO_ID, (_onnx_path,)),
    ]
    _onnx_input_names = ("x",)
    _onnx_output_names = ("fetch_name_0",)

    _transformers_path: ClassVar[str] = "PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors"
    _transformers_download_options: ClassVar[list[DownloadOption]] = [
        DownloadOption(DownloadPlatform.HUGGINGFACE, _transformers_path),
    ]

    _default_runtime = InferenceRuntime.OPENCV

    def __init__(
        self,
        runtime: InferenceRuntime = _default_runtime,
        model_path: Path | str | None = None,
    ) -> None:
        match runtime:
            case InferenceRuntime.ONNXRUNTIME:
                self._onnxruntime_init(model_path)
                self._run_fn = self._onnxruntime_run
            case InferenceRuntime.OPENCV:
                self._opencv_init(model_path)
                self._run_fn = self._opencv_run
            case InferenceRuntime.TRANSFORMERS:
                self._transformers_init(model_path)
                self._run_fn = self._transformers_run

    @override
    def _transformers_init(self, model_path: Path | str | None = None) -> None:
        super()._transformers_init(model_path=model_path)

        from transformers import PPLCNetForImageClassification, PPLCNetImageProcessor

        self._transformers_model = PPLCNetForImageClassification.from_pretrained(
            self._transformers_path
        )
        self._transformers_processor = PPLCNetImageProcessor.from_pretrained(
            self._transformers_path
        )

    def __call__(self, input: Iterable[NDArray[np.uint8]]) -> list[Classification]:
        return self._run_fn(input)

    def _onnx_preprocess(self, input: Iterable[NDArray[np.uint8]]) -> NDArray:
        """PP-LCNet image preprocessing pipeline.

        Args:
            input: iterable of HxWxC uint8 images (C=3, assumed BGR).

        Output:
            list of CxHxW float32 tensors (BGR order), normalized with PP-LCNet mean/std.
        """
        resize_short = 256  # shorter edge after resize
        crop_size = 224  # center crop size

        cropped_imgs: list[NDArray] = []

        scalefactor = (
            1.0 / (255.0 * 0.229),  # B
            1.0 / (255.0 * 0.224),  # G
            1.0 / (255.0 * 0.225),  # R
        )
        mean = (
            0.485 * 255.0,  # B
            0.456 * 255.0,  # G
            0.406 * 255.0,  # R
        )

        params = cv2.dnn.Image2BlobParams(
            scalefactor=scalefactor,
            mean=mean,
            swapRB=False,
            ddepth=cv2.CV_32F,
            datalayout=cv2.DNN_LAYOUT_NCHW,
        )

        for img in input:
            # Validate and coerce to expected dtype/layout (HWC, uint8, 3 channels)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 image, got shape={img.shape}")
            if img.dtype != np.uint8:
                raise ValueError(f"Expected uint8 image, got dtype={img.dtype}")

            # Resize while preserving aspect ratio using the shorter edge as reference
            h, w = img.shape[:2]
            scale = resize_short / min(h, w)
            new_size = (round(w * scale), round(h * scale))
            resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

            # Center-crop
            top = (new_size[1] - crop_size) // 2
            left = (new_size[0] - crop_size) // 2
            cropped = resized[top : top + crop_size, left : left + crop_size, :]

            cropped_imgs.append(cropped)

        return cv2.dnn.blobFromImagesWithParams(cropped_imgs, params)

    @classmethod
    def _postprocess(cls, pred: Sequence[Sequence[np.float32]]) -> list[Classification]:
        return [Classification(max(p), int(np.argmax(p))) for p in pred]

    def _onnxruntime_run(self, input: Iterable[NDArray[np.uint8]]) -> list[Classification]:
        logger.debug("Started preprocessing")
        images = self._onnx_preprocess(input)

        input_dict = dict(zip(self._onnx_input_names, [images]))

        logger.debug("Done preprocessing")
        logger.debug("Started running the model")

        output = self._onnxruntime_session.run(self._onnx_output_names, input_dict)[0]

        logger.debug("Done running the model")
        logger.debug("Started postprocessing")

        result = self._postprocess(output)  # type: ignore

        logger.debug("Done postprocessing")

        return result

    def _opencv_run(self, input: Iterable[NDArray[np.uint8]]) -> list[Classification]:
        logger.debug("Started preprocessing")

        images = self._onnx_preprocess(input)
        self._opencv_net.setInput(images)

        logger.debug("Done preprocessing")
        logger.debug("Started running the model")

        output = self._opencv_net.forward()

        logger.debug("Done running the model")
        logger.debug("Started postprocessing")

        result = self._postprocess(output)  # type: ignore

        logger.debug("Done postprocessing")

        return result

    def _transformers_run(self, input: Iterable[NDArray[np.uint8]]) -> list[Classification]:
        logger.debug("Started preprocessing")

        import torch.nn.functional as F

        inputs = self._transformers_processor(
            images=input,  # ty:ignore[invalid-argument-type]
            return_tensors="pt",
        ).to(self._transformers_model.device)

        logger.debug("Done preprocessing")
        logger.debug("Started running the model")

        output = self._transformers_model(**inputs)

        logger.debug("Done running the model")
        logger.debug("Started postprocessing")

        logits = output.last_hidden_state

        probs = F.softmax(logits, dim=-1).detach().numpy()

        result = self._postprocess(probs)

        logger.debug("Done postprocessing")

        return result
