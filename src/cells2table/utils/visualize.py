import cv2
import numpy as np
from numpy.typing import NDArray

from .table import Table


def visualize_table(
    image: NDArray[np.uint8],
    table: Table,
    color=(0, 255, 0),
    thickness=2,
    window_name="Bounding Boxes",
):
    """
    image: np.ndarray (BGR image loaded with cv2)
    """

    img = image.copy()

    for cell in table.cells:
        cv2.rectangle(
            img,
            (int(cell.bbox.l), int(cell.bbox.t)),
            (int(cell.bbox.r), int(cell.bbox.b)),
            color,
            thickness,
        )

    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
