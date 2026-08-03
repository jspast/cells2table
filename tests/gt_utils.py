import json
from collections.abc import Iterator
from os import getenv
from pathlib import Path

from cells2table.models.tasks import ClassificationResult, DetectionResult

UPDATE_GT = bool(int(getenv("CELLS2TABLE_UPDATE_GT", "0")))

# Absolute tolerance for confidence results
CONFIDENCE_TOLERANCE = 0.01

# Absolute tolerance for positional results in pixels
POSITION_TOLERANCE = 5


def verify_text(gt_file_path: Path, text: str, update_gt: bool = UPDATE_GT) -> None:
    if update_gt:
        gt_file_path.parent.mkdir(parents=True, exist_ok=True)
        gt_file_path.write_text(text, encoding="utf-8")

    else:
        gt_text = gt_file_path.read_text(encoding="utf-8")
        assert text == gt_text, f"Ground-truth:\n{gt_text}\n\nResult:\n{text}"


def verify_classification(
    gt_file_path: Path,
    result: ClassificationResult,
    update_gt: bool = UPDATE_GT,
    key: str = "classification",
) -> None:
    if update_gt:
        gt_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            gt_dict = json.loads(gt_file_path.read_text())
        except FileNotFoundError:
            gt_dict = {}

        res = {"class": result.cls, "confidence": round(float(result.confidence), 4)}

        gt_dict[key] = res
        gt_file_path.write_text(json.dumps(gt_dict), encoding="utf-8")

    else:
        gt_dict = json.loads(gt_file_path.read_text())

        gt = gt_dict[key]

        assert gt["class"] == result.cls
        assert abs(gt["confidence"] - result.confidence) < CONFIDENCE_TOLERANCE


def verify_detection(
    gt_file_path: Path,
    result: Iterator[DetectionResult],
    update_gt: bool = UPDATE_GT,
    key: str = "detection",
) -> None:
    if update_gt:
        gt_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            gt_dict = json.loads(gt_file_path.read_text())
        except FileNotFoundError:
            gt_dict = {}

        res = [
            {
                "bbox": {
                    "l": round(float(r.bbox[0]), 2),
                    "t": round(float(r.bbox[1]), 2),
                    "r": round(float(r.bbox[2]), 2),
                    "b": round(float(r.bbox[3]), 2),
                },
                "confidence": round(float(r.confidence), 4),
            }
            for r in sorted(result, key=lambda d: d.bbox[0] + 100 * d.bbox[1])
        ]

        gt_dict[key] = res
        gt_file_path.write_text(json.dumps(gt_dict), encoding="utf-8")

    else:
        gt_dict = json.loads(gt_file_path.read_text())

        res = sorted(result, key=lambda d: d.bbox[0] + 100 * d.bbox[1])

        for i, r in enumerate(res):
            gt = gt_dict[key][i]

            assert abs(gt["bbox"]["l"] - r.bbox[0]) < POSITION_TOLERANCE
            assert abs(gt["bbox"]["t"] - r.bbox[1]) < POSITION_TOLERANCE
            assert abs(gt["bbox"]["r"] - r.bbox[2]) < POSITION_TOLERANCE
            assert abs(gt["bbox"]["b"] - r.bbox[3]) < POSITION_TOLERANCE
            assert abs(gt["confidence"] - r.confidence) < CONFIDENCE_TOLERANCE
