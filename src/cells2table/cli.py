import argparse
from pathlib import Path

import cv2

from .models.PaddlePaddle import PaddlePaddleTablePipeline
from .utils.visualize import visualize_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Load an image from a given path using OpenCV")
    parser.add_argument("image_path", type=Path, help="Path to the image file")

    args = parser.parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"File does not exist: {args.image_path}")

    image = cv2.imread(str(args.image_path))

    if image is None:
        raise ValueError(f"Failed to load image: {args.image_path}")

    print("Image loaded successfully")
    print(f"Shape: {image.shape} (H, W, C)")
    print(f"Data type: {image.dtype}")

    table_pipeline = PaddlePaddleTablePipeline()
    tables = table_pipeline([image])  # type: ignore

    for table in tables:
        visualize_table(image, table)  # type: ignore


if __name__ == "__main__":
    main()
