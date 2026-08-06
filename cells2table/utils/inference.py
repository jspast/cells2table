import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class InferenceRuntime(StrEnum):
    ONNXRUNTIME = "onnxruntime"
    OPENCV = "opencv"
    TRANSFORMERS = "transformers"


enabled_inference_runtimes: list[InferenceRuntime] = [InferenceRuntime.OPENCV]
