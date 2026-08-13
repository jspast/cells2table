import argparse
import logging
import os
import platform
from pathlib import Path

import cv2

from cells2table.models.PaddlePaddle import PaddlePaddleLayoutModel
from cells2table.pipelines import DefaultTablePipeline
from cells2table.utils.inference import enabled_inference_runtimes
from cells2table.utils.visualize import show_image, visualize_detections, visualize_table

logger = logging.getLogger(__name__)


def fix_cv2_qt_warnings() -> None:
    if platform.system() == "Linux":
        os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        os.environ["XDG_SESSION_TYPE"] = "x11"


def setup_logging(level: logging._Level = logging.INFO) -> None:
    format = "%(asctime)s\t%(levelname)s\t%(name)s: %(message)s"
    logging.basicConfig(level=level, format=format)


def download() -> None:
    """Download default pipeline models."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=None, help="Path to download models to")
    parser.add_argument("-log", "--loglevel", default=logging.INFO, help="Logging level to use")

    args = parser.parse_args()

    setup_logging(args.loglevel)

    DefaultTablePipeline.download(
        runtime=DefaultTablePipeline._default_runtime, local_dir=args.local_dir
    )


def main() -> None:
    """Basic CLI program for testing."""

    fix_cv2_qt_warnings()

    parser = argparse.ArgumentParser(description="Load an image from a given path using OpenCV")
    parser.add_argument("image_path", type=Path, help="Path to the image file")
    parser.add_argument(
        "--task", type=str, default="table", help="The task to perform", choices=["table", "layout"]
    )
    parser.add_argument("--models-path", type=Path, default=None, help="Path to downloaded models")
    parser.add_argument(
        "--runtime",
        type=str,
        default=None,
        help="Inference runtime to use",
        choices=enabled_inference_runtimes,
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default=logging.INFO,
        help="Logging level to use",
        choices=logging.getLevelNamesMapping().keys(),
    )

    args = parser.parse_args()

    setup_logging(args.loglevel)

    if not args.image_path.exists():
        raise FileNotFoundError(f"File does not exist: {args.image_path}")

    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {args.image_path}")

    logger.info("Image loaded successfully from %s", args.image_path)
    logger.debug(
        "Image proprieties: width=%d, height=%d, channels=%d, datatype=%s",
        image.shape[1],
        image.shape[0],
        image.shape[2],
        str(image.dtype),
    )

    if args.task[0].upper() == "T":
        table_pipeline = DefaultTablePipeline(args.models_path, args.runtime)
        tables = table_pipeline([image])

        for table in tables:
            show_image(visualize_table(image, table))  # ty:ignore[invalid-argument-type]

    elif args.task[0].upper() == "L":
        layout_model = PaddlePaddleLayoutModel(args.models_path, args.runtime)
        layouts = layout_model([image])  # ty: ignore[invalid-argument-type]

        for detections in layouts:
            show_image(visualize_detections(image, detections, layout_model.id2label))  # ty:ignore[invalid-argument-type]

    else:
        raise ValueError(f"Unrecognized task '{args.task}'")


if __name__ == "__main__":
    main()
