import logging
from typing import Iterable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from cells2table.models.runtimes.onnx import OnnxModel
from cells2table.models.tasks import ClassificationModel, ClassificationResult
from cells2table.utils.download import DownloadOptions, DownloadPlatform

HF_REPO_ID = "jspast/paddlepaddle-table-models-onnx"

logger = logging.getLogger(__name__)


class PaddlePaddleTableClassificationModel(ClassificationModel, OnnxModel):
    classes = ["wired", "wireless"]

    @classmethod
    def get_onnx_path(cls) -> str:
        return "table_cls.onnx"

    @classmethod
    def get_download_options(cls) -> DownloadOptions:
        return DownloadOptions(DownloadPlatform.HUGGINGFACE, HF_REPO_ID, [cls.get_onnx_path()])

    def __call__(self, input: Iterable[NDArray[np.uint8]]) -> list[ClassificationResult]:
        logger.debug("Started preprocessing")
        images = self.preprocess(input)

        input_dict = dict(zip(self.input_names, [images]))

        logger.debug("Done preprocessing")
        logger.debug("Started running the model")

        output = self.session.run(self.output_names, input_dict)[0]

        logger.debug("Done running the model")
        logger.debug("Started postprocessing")

        result = self.postprocess(output)  # type: ignore

        logger.debug("Done postprocessing")

        return result

    def preprocess(self, input: Iterable[NDArray[np.uint8]]) -> list[NDArray[np.float32]]:
        """PP-LCNet image preprocessing pipeline.

        Args:
            input: iterable of HxWxC uint8 images (C=3, assumed RGB).

        Output:
            list of CxHxW float32 tensors (BGR order), normalized with PP-LCNet mean/std.
        """
        resize_short = 256  # shorter edge after resize
        crop_size = 224  # center crop size
        mean = np.asarray([0.406, 0.456, 0.485], dtype=np.float32)  # RGB mean
        std = np.asarray([0.225, 0.224, 0.229], dtype=np.float32)  # RGB std
        rescale_factor = 1.0 / 255.0  # uint8 -> [0,1]

        out: list[NDArray[np.float32]] = []

        for img in input:
            # Validate and coerce to expected dtype/layout (HWC, uint8, 3 channels)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 image, got shape={img.shape}")
            if img.dtype != np.uint8:
                raise ValueError(f"Expected uint8 image, got dtype={img.dtype}")

            h, w = img.shape[:2]

            # Resize while preserving aspect ratio using the shorter edge as reference
            scale = resize_short / float(min(h, w))
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))

            # Perform the resize (OpenCV expects size as (width, height))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Center-crop to crop_size x crop_size (assumes resized dims are >= crop_size)
            if new_h < crop_size or new_w < crop_size:
                raise ValueError(
                    f"Resized image too small for center crop: resized={new_h}x{new_w}, crop={crop_size}"
                )
            top = (new_h - crop_size) // 2
            left = (new_w - crop_size) // 2
            cropped = resized[top : top + crop_size, left : left + crop_size, :]

            # Convert to float32 and rescale to [0,1]
            x = cropped.astype(np.float32) * rescale_factor

            # Normalize per channel in RGB space: (x - mean) / std
            x = (x - mean) / std

            # Convert RGB -> BGR
            x = x[..., ::-1]

            # Convert HWC -> CHW
            x = np.transpose(x, (2, 0, 1)).astype(np.float32, copy=False)

            out.append(x)

        return out

    @classmethod
    def postprocess(cls, pred: Sequence[Sequence[float]]) -> list[ClassificationResult]:
        return [ClassificationResult(cls.classes[np.argmax(p)], max(p)) for p in pred]
