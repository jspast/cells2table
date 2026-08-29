import logging
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, override

import cv2
import numpy as np
from numpy.typing import NDArray

from cells2table.models.runtimes import ONNXRuntimeModel, OpenCVModel, TransformersModel
from cells2table.models.tasks import Classification, ClassificationModel
from cells2table.utils.download import DownloadOption, DownloadPlatform
from cells2table.utils.inference import InferenceRuntime

logger = logging.getLogger(__name__)


class PaddlePaddleTableClassificationModel(
    ClassificationModel, ONNXRuntimeModel, OpenCVModel, TransformersModel
):
    id2label: ClassVar[dict[int, str]] = {0: "wired", 1: "wireless"}

    _input_shape = (224, 224)

    _onnx_repo: ClassVar[str] = "jspast/paddlepaddle-table-models-onnx"
    _onnx_path: ClassVar[str] = "table_cls.onnx"
    _onnx_download_options: ClassVar[list[DownloadOption]] = [
        DownloadOption(DownloadPlatform.HUGGINGFACE, _onnx_repo, (_onnx_path,)),
    ]
    _onnx_input_names = ("x",)
    _onnx_output_names = ("fetch_name_0",)

    _transformers_repo: ClassVar[str] = "PaddlePaddle/PP-LCNet_x1_0_table_cls_safetensors"
    _transformers_download_options: ClassVar[list[DownloadOption]] = [
        DownloadOption(DownloadPlatform.HUGGINGFACE, _transformers_repo),
    ]

    def __init__(
        self,
        model_path: Path | str | None = None,
        runtime: InferenceRuntime | None = None,
    ) -> None:
        super().__init__(model_path, runtime)

        match self._runtime:
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

        self._transformers_model = PPLCNetForImageClassification.from_pretrained(self.model_path)
        self._transformers_processor = PPLCNetImageProcessor.from_pretrained(self.model_path)

    def __call__(self, input: Sequence[NDArray[np.uint8]]) -> list[Classification]:
        return self._run_fn(input)

    @classmethod
    def _onnx_preprocess(cls, input: Sequence[NDArray[np.uint8]]) -> NDArray:
        """PP-LCNet image preprocessing pipeline.

        Args:
            input: Sequence of HxWxC uint8 images (C=3, assumed BGR).

        Output:
            list of CxHxW float32 tensors (BGR order), normalized with PP-LCNet mean/std.
        """
        resize_short = 256  # shorter edge after resize
        crop_size = 224  # center crop size

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        out = np.empty((len(input), 3, crop_size, crop_size), dtype=np.float32)

        for i, img in enumerate(input):
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
            cropped = resized[top : top + crop_size, left : left + crop_size]

            out[i] = (
                cropped.transpose(2, 0, 1).astype(np.float32) / 255.0 - mean[:, None, None]
            ) / std[:, None, None]

        return out

    @classmethod
    def _postprocess(cls, pred: Sequence[Sequence[np.float32]]) -> list[Classification]:
        return [Classification(max(p), int(np.argmax(p))) for p in pred]

    def _onnxruntime_run(self, input: Sequence[NDArray[np.uint8]]) -> list[Classification]:
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

    def _opencv_run(self, input: Sequence[NDArray[np.uint8]]) -> list[Classification]:
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

    def _transformers_run(self, input: Sequence[NDArray[np.uint8]]) -> list[Classification]:
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
