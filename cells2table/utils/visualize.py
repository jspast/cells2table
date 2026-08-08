from collections.abc import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from cells2table.datamodels import Table
from cells2table.models.tasks import ClassifiedDetection, Detection


def bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ty:ignore[invalid-return-type]


def rgb_to_bgr(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # ty:ignore[invalid-return-type]


def show_image(image: NDArray[np.uint8], window_title: str = "Image") -> None:
    """Create a window to show an image.

    Args:
        image: A cv2 BGR image.
        window_title: The title of the created window.
    """
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_table(
    image: NDArray[np.uint8],
    table: Table,
    color=(0, 255, 0),
    thickness=2,
) -> NDArray[np.uint8]:
    """Simple table cells visualization on top of the image.

    The Row, Col, Row Span and Col Span will be printed for each cell.
    The format is `R,C : RS,CS`.

    Args:
        image: A cv2 BGR image of the table.
        table: The table to draw on top of the image.
        color: The color of the table overlay.
        thickness: The thickness of the overlay lines.
    """
    img = image.copy()

    for cell in table.cells:
        cv2.rectangle(
            img,
            (round(cell.bbox.l), round(cell.bbox.t)),
            (round(cell.bbox.r), round(cell.bbox.b)),
            color,
            thickness,
        )

        cv2.putText(
            img,
            f"{cell.row},{cell.col} : {cell.row_span},{cell.col_span}",
            (round(cell.bbox.l), round(cell.bbox.t) + 10),
            (128, 192, 0),
            cv2.FontFace("uni"),
            14,
        )

    return img


def visualize_detections(
    image: NDArray[np.uint8],
    detections: Iterator[Detection],
    id2label: dict[int, str] | None = None,
    color=(0, 255, 0),
    thickness=2,
) -> NDArray[np.uint8]:
    """Simple detection visualization on top of the image.

    In case detections are classified, the label will be printed for each.

    Args:
        image: A cv2 BGR image.
        detections: The detections to draw on top of the image.
        color: The color of the detection overlays.
        thickness: The thickness of the overlay lines.
    """
    img = image.copy()

    for det in detections:
        cv2.rectangle(
            img,
            (round(det.bbox[0]), round(det.bbox[1])),
            (round(det.bbox[2]), round(det.bbox[3])),
            color,
            thickness,
        )

        if isinstance(det, ClassifiedDetection) and id2label is not None:
            cv2.putText(
                img,
                id2label[det.id],
                (round(det.bbox[0]), round(det.bbox[1]) + 10),
                (128, 192, 0),
                cv2.FontFace("uni"),
                14,
            )

    return img
