import logging
from enum import StrEnum

logger = logging.getLogger(__name__)


class InferenceRuntime(StrEnum):
    ONNXRUNTIME = "onnxruntime"
    OPENCV = "opencv"
    TRANSFORMERS = "transformers"


enabled_inference_runtimes: set[InferenceRuntime] = {InferenceRuntime.OPENCV}


def select_inference_runtime(supported: list[InferenceRuntime]) -> InferenceRuntime:
    logger.debug("Enabled inference runtimes: %s", enabled_inference_runtimes)
    logger.debug("Supported inference runtimes: %s", supported)
    for s in supported:
        if s in enabled_inference_runtimes:
            logger.info("Automatically selected inference runtime '%s'", s)
            return s

    raise ValueError("No supported inference runtime found. Check enabled_inference_runtimes.")
