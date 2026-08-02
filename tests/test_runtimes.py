import json
from pathlib import Path

import cv2
import pytest
from numpy.typing import NDArray

from cells2table.pipelines import DefaultPipeline
from cells2table.pipelines.classification_detection import ClassificationDetectionPipeline
from cells2table.utils.inference import InferenceRuntime


@pytest.fixture
def opencv_pipeline() -> ClassificationDetectionPipeline:
    return DefaultPipeline(runtime=InferenceRuntime.OPENCV)


@pytest.fixture
def onnx_pipeline() -> ClassificationDetectionPipeline:
    return DefaultPipeline(runtime=InferenceRuntime.ONNX)


@pytest.fixture
def test_file_path() -> Path:
    return Path(__file__).parent / "data" / "images" / "wired.png"


@pytest.fixture
def test_image(test_file_path: Path) -> NDArray:
    return cv2.imread(test_file_path)  # ty:ignore[invalid-return-type]


@pytest.fixture
def gt_file_path() -> Path:
    return Path(__file__).parent / "data" / "gt" / "wired.json"


@pytest.fixture
def gt_dict(gt_file_path: Path) -> dict:
    return json.loads(gt_file_path.read_text())


def test_opencv_classification(
    opencv_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_dict: dict,
) -> None:
    result = opencv_pipeline.classification_model([test_image])

    assert gt_dict["classification"] == eval(json.dumps(str(result)))


def test_opencv_detection_wired(
    opencv_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_dict: dict,
) -> None:
    model = opencv_pipeline.detection_models[0]
    result = model([test_image])

    assert gt_dict["detection_wired"] == eval(json.dumps(str(list(result[0]))))


def test_opencv_detection_wireless(
    opencv_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_dict: dict,
) -> None:
    model = opencv_pipeline.detection_models[1]
    result = model([test_image])

    assert gt_dict["detection_wireless"] == eval(json.dumps(str(list(result[0]))))


def test_onnx_classification(
    onnx_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_dict: dict,
) -> None:
    result = onnx_pipeline.classification_model([test_image])

    assert gt_dict["classification"] == eval(json.dumps(str(result)))


def test_onnx_detection_wired(
    onnx_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_dict: dict,
) -> None:
    model = onnx_pipeline.detection_models[0]
    result = model([test_image])

    assert gt_dict["detection_wired"] == eval(json.dumps(str(list(result[0]))))


def test_onnx_detection_wireless(
    onnx_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_dict: dict,
) -> None:
    model = onnx_pipeline.detection_models[1]
    result = model([test_image])

    assert gt_dict["detection_wireless"] == eval(json.dumps(str(list(result[0]))))
