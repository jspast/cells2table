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
) -> None:
    """Simple table visualization on top of the image.

    image: np.ndarray (BGR image loaded with cv2)
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
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 192, 0),
            thickness,
        )

    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
