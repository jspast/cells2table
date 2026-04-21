import cv2
import numpy as np
from numpy.typing import NDArray

from cells2table.datamodels import Table
from cells2table.models.tasks.detection import DetectionResult


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
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (128, 192, 0),
            thickness,
        )

    return img


def visualize_detections(
    image: NDArray[np.uint8],
    detections: list[DetectionResult],
) -> None:
    """Detections visualization on top of the image.

    Args:
        image: A cv2 image of the table.
        table: The table to draw on top of the image.
        color: The color of the table overlay.
        thickness: The thickness of the overlay lines.
    """

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RangeSlider

    img = bgr_to_rgb(image.copy())

    # Normalize + colormap
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap("winter")

    # Create figure
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(img)
    ax.axis("off")

    # Store rectangle artists
    rects = []
    texts = []

    for bbox, conf in detections:
        color = cmap(norm(conf))
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        rects.append((rect, conf))

        txt = ax.text(bbox[0], bbox[1] - 5, f"{conf:.2f}", color=color)
        texts.append((txt, conf))

    # Add range slider
    slider_ax = plt.axes((0.2, 0.1, 0.6, 0.03))
    slider = RangeSlider(slider_ax, "Confidence", 0.0, 1.0, valinit=(0.5, 1.0))

    # Update function
    def update(val):
        min_conf, max_conf = slider.val

        for rect, conf in rects:
            rect.set_visible(min_conf <= conf <= max_conf)

        for txt, conf in texts:
            txt.set_visible(min_conf <= conf <= max_conf)

        fig.canvas.draw_idle()

    update(slider.val)
    slider.on_changed(update)

    plt.show()
