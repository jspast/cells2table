from pathlib import Path

import cv2
import pytest
from numpy.typing import NDArray

from cells2table.models.PaddlePaddle import PaddlePaddleLayoutModel
from cells2table.models.tasks import ClassifiedDetectionModel
from cells2table.utils.inference import InferenceRuntime
from tests.gt_utils import verify_classifieddetection


@pytest.fixture
def onnxruntime_model() -> ClassifiedDetectionModel:
    return PaddlePaddleLayoutModel(runtime=InferenceRuntime.ONNXRUNTIME)


@pytest.fixture
def transformers_model() -> ClassifiedDetectionModel:
    return PaddlePaddleLayoutModel(runtime=InferenceRuntime.TRANSFORMERS)


@pytest.fixture
def test_file_path() -> Path:
    return Path(__file__).parent / "data" / "images" / "layout.png"


@pytest.fixture
def test_image(test_file_path: Path) -> NDArray:
    return cv2.imread(test_file_path)  # ty:ignore[invalid-return-type]


@pytest.fixture
def gt_file_path() -> Path:
    return Path(__file__).parent / "data" / "gt" / "layout.json"


def test_onnxruntime(
    onnxruntime_model: ClassifiedDetectionModel,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    model = onnxruntime_model
    result = model([test_image])
    verify_classifieddetection(gt_file_path, result[0], key="detection")


def test_transformers_detection_wired(
    transformers_model: ClassifiedDetectionModel,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    model = transformers_model
    result = model([test_image])
    verify_classifieddetection(gt_file_path, result[0], False, key="detection")
