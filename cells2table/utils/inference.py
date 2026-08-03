import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class InferenceRuntime(StrEnum):
    ONNX = "onnxruntime"
    OPENCV = "opencv"
    TRANSFORMERS = "transformers"
