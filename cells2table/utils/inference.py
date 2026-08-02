import logging
from enum import Enum

logger = logging.getLogger(__name__)


class InferenceRuntime(Enum):
    ONNX = "onnx"
    OPENCV = "opencv"
