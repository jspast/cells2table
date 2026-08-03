from pathlib import Path

import cv2
import pytest
from numpy.typing import NDArray

from cells2table.pipelines import DefaultPipeline
from cells2table.pipelines.classification_detection import ClassificationDetectionPipeline
from cells2table.utils.inference import InferenceRuntime
from tests.gt_utils import verify_classification, verify_detection


@pytest.fixture
def opencv_pipeline() -> ClassificationDetectionPipeline:
    return DefaultPipeline(runtime=InferenceRuntime.OPENCV)


@pytest.fixture
def onnx_pipeline() -> ClassificationDetectionPipeline:
    return DefaultPipeline(runtime=InferenceRuntime.ONNX)


@pytest.fixture
def transformers_pipeline() -> ClassificationDetectionPipeline:
    return DefaultPipeline(runtime=InferenceRuntime.TRANSFORMERS)


@pytest.fixture
def test_file_path() -> Path:
    return Path(__file__).parent / "data" / "images" / "wired.png"


@pytest.fixture
def test_image(test_file_path: Path) -> NDArray:
    return cv2.imread(test_file_path)  # ty:ignore[invalid-return-type]


@pytest.fixture
def gt_file_path() -> Path:
    return Path(__file__).parent / "data" / "gt" / "wired.json"


def test_opencv_classification(
    opencv_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    result = opencv_pipeline.classification_model([test_image])
    verify_classification(gt_file_path, result[0])


def test_opencv_detection_wired(
    opencv_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    model = opencv_pipeline.detection_models[0]
    result = model([test_image])
    verify_detection(gt_file_path, result[0], key="detection_wired")


def test_opencv_detection_wireless(
    opencv_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    model = opencv_pipeline.detection_models[1]
    result = model([test_image])
    verify_detection(gt_file_path, result[0], key="detection_wireless")


def test_onnx_classification(
    onnx_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    result = onnx_pipeline.classification_model([test_image])
    verify_classification(gt_file_path, result[0], False)


def test_onnx_detection_wired(
    onnx_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    model = onnx_pipeline.detection_models[0]
    result = model([test_image])
    verify_detection(gt_file_path, result[0], False, key="detection_wired")


def test_onnx_detection_wireless(
    onnx_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    model = onnx_pipeline.detection_models[1]
    result = model([test_image])
    verify_detection(gt_file_path, result[0], False, key="detection_wireless")


def test_transformers_classification(
    onnx_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    result = onnx_pipeline.classification_model([test_image])
    verify_classification(gt_file_path, result[0], False)


def test_transformers_detection_wired(
    transformers_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    model = transformers_pipeline.detection_models[0]
    result = model([test_image])
    verify_detection(gt_file_path, result[0], False, key="detection_wired")


def test_transformers_detection_wireless(
    transformers_pipeline: ClassificationDetectionPipeline,
    test_image: NDArray,
    gt_file_path: Path,
) -> None:
    model = transformers_pipeline.detection_models[1]
    result = model([test_image])
    verify_detection(gt_file_path, result[0], False, key="detection_wireless")
