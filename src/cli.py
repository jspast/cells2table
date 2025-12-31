import argparse
import logging
from pathlib import Path

import cv2

from cells2table import DefaultPipeline
from cells2table.utils.visualize import visualize_table

logger = logging.getLogger(__name__)


def main() -> None:
    """Basic CLI program for testing."""

    log_format = "%(asctime)s\t%(levelname)s\t%(name)s: %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=log_format)

    parser = argparse.ArgumentParser(description="Load an image from a given path using OpenCV")
    parser.add_argument("image_path", type=Path, help="Path to the image file")

    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"File does not exist: {args.image_path}")

    image = cv2.imread(str(args.image_path))

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

    table_pipeline = DefaultPipeline()
    tables = table_pipeline([image])  # type: ignore

    for table in tables:
        visualize_table(image, table)  # type: ignore


if __name__ == "__main__":
    main()
