import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Annotated, ClassVar, Literal

import numpy as np

try:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import Cluster, LayoutPrediction, Page
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import BaseLayoutOptions
    from docling.datamodel.settings import settings
    from docling.models.base_layout_model import BaseLayoutModel
    from docling.utils.profiling import TimeRecorder
    from docling.utils.visualization import draw_clusters_and_cells_side_by_side
    from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel
    from pydantic import Field
except ImportError:
    raise ImportError("docling is not installed. Unable to initialize plugin.")

from cells2table.models.PaddlePaddle import PaddlePaddleLayoutModel
from cells2table.models.tasks import ClassifiedDetection
from cells2table.utils.inference import InferenceRuntime

_log = logging.getLogger(__name__)


class CustomDoclingLayoutOptions(BaseLayoutOptions):
    kind: ClassVar[Literal["ppdoclayoutv3"]] = "ppdoclayoutv3"

    confidence_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum confidence score to keep a cell detection.",
        ),
    ] = Field(
        default_factory=lambda: float(os.environ.get("PPDOCLAYOUTV3_CONFIDENCE_THRESHOLD", "0.5"))
    )

    runtime: Annotated[
        InferenceRuntime, Field(description="Inference runtime to use. Defaults to ONNX Runtime.")
    ] = InferenceRuntime.ONNXRUNTIME


class CustomDoclingLayoutModel(BaseLayoutModel):
    id2label: ClassVar[dict[int, DocItemLabel]] = {
        0: DocItemLabel.TEXT,
        1: DocItemLabel.CODE,
        2: DocItemLabel.TEXT,
        3: DocItemLabel.PICTURE,
        4: DocItemLabel.TEXT,
        5: DocItemLabel.FORMULA,
        6: DocItemLabel.TITLE,
        7: DocItemLabel.CAPTION,
        8: DocItemLabel.PAGE_FOOTER,
        9: DocItemLabel.PAGE_FOOTER,
        10: DocItemLabel.FOOTNOTE,
        11: DocItemLabel.TEXT,
        12: DocItemLabel.PAGE_HEADER,
        13: DocItemLabel.PAGE_HEADER,
        14: DocItemLabel.PICTURE,
        15: DocItemLabel.FORMULA,
        16: DocItemLabel.TEXT,
        17: DocItemLabel.SECTION_HEADER,
        18: DocItemLabel.TEXT,
        19: DocItemLabel.TEXT,
        20: DocItemLabel.PICTURE,
        21: DocItemLabel.TABLE,
        22: DocItemLabel.TEXT,
        23: DocItemLabel.TEXT,
        24: DocItemLabel.FOOTNOTE,
    }

    requires_layout_postprocessing: bool = True

    def __init__(
        self,
        artifacts_path: Path | None,
        accelerator_options: AcceleratorOptions,
        options: CustomDoclingLayoutOptions,
        enable_remote_services: bool = False,
        **kwargs,
    ) -> None:
        self.options = options

        if artifacts_path is None:
            models_path = None
        # elif (artifacts_path / DefaultTablePipeline._onnx_dirname).exists():
        #     models_path = artifacts_path / DefaultTablePipeline._onnx_dirname
        else:
            models_path = artifacts_path

        self.model = PaddlePaddleLayoutModel(options.runtime, models_path)

    @classmethod
    def get_options_type(cls) -> type[CustomDoclingLayoutOptions]:
        return CustomDoclingLayoutOptions

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:

        pages = list(pages)

        page_images: list[np.ndarray] = []
        image_page: list[int] = []

        for i, page in enumerate(pages):
            assert page._backend is not None
            if not page._backend.is_valid():
                continue

            # TODO: Ideally we should get the image with it's min size equal to the model input
            page_image = page.get_image(scale=1.5)
            if page_image is None:
                continue

            page_images.append(np.array(page_image, copy=True))
            image_page.append(i)

        if len(page_images) == 0:
            return []

        with TimeRecorder(conv_res, "layout"):
            output = self.model(page_images, conf_threshold=self.options.confidence_threshold)

        predictions: list[LayoutPrediction] = [
            page.predictions.layout or LayoutPrediction() for page in pages
        ]

        for page_id, image, out in zip(image_page, page_images, output):
            page = pages[page_id]
            clusters = self._predictions_to_clusters(page=page, image=image, detections=out)

            if settings.debug.visualize_raw_layout:
                draw_clusters_and_cells_side_by_side(
                    conv_res.input.file, page, clusters, mode_prefix="raw"
                )

            # Emit raw clusters; post-processing and layout_score are
            # handled by the downstream LayoutPostprocessingModel stage.
            predictions[page_id] = LayoutPrediction(clusters=clusters)

        for page, prediction in zip(pages, predictions):
            page.predictions.layout = prediction

        return predictions

    def _predictions_to_clusters(
        self,
        page: Page,
        image: np.ndarray,
        detections: Iterator[ClassifiedDetection],
    ) -> list[Cluster]:
        assert page.size is not None
        page_width = page.size.width
        page_height = page.size.height
        scale_x = page_width / image.shape[1]
        scale_y = page_height / image.shape[0]

        clusters: list[Cluster] = []
        for i, det in enumerate(detections):
            label = self.id2label.get(det.id)
            if label is None:
                _log.warning(
                    "Dropping detections with label id %s: the model emitted an "
                    "id that is absent from its own config id2label map.",
                    det.id,
                )
                continue

            # Detections can overshoot the page; downstream area ratios in
            # LayoutPostprocessor assume page-bounded boxes.
            bbox = BoundingBox(
                l=min(max(det.bbox[0] * scale_x, 0.0), page_width),
                t=min(max(det.bbox[1] * scale_y, 0.0), page_height),
                r=min(max(det.bbox[2] * scale_x, 0.0), page_width),
                b=min(max(det.bbox[3] * scale_y, 0.0), page_height),
                coord_origin=CoordOrigin.TOPLEFT,
            )
            clusters.append(
                Cluster(
                    id=i,
                    label=label,
                    confidence=float(det.confidence),
                    bbox=bbox,
                    cells=[],
                )
            )
        return clusters


# Plugin factory
def layout_engines():
    return {"layout_engines": [CustomDoclingLayoutModel]}
