import os
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Sequence, Type

import numpy

try:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import Cluster, Page, Table, TableStructurePrediction
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import BaseTableStructureOptions
    from docling.datamodel.settings import settings
    from docling.models.base_table_model import BaseTableStructureModel
    from docling.utils.profiling import TimeRecorder
    from docling_core.types.doc.base import BoundingBox
    from docling_core.types.doc.document import TableCell
    from docling_core.types.doc.labels import DocItemLabel
    from PIL import ImageDraw
    from pydantic import Field
except ImportError:
    raise ImportError("docling is not installed. Unable to initialize plugin.")

from cells2table.datamodels import Table as PluginTable
from cells2table.pipelines import DefaultPipeline


def build_docling_table(
    table: PluginTable,
    table_cluster: Cluster,
    page: Page,
    scale: float,
) -> Table:
    docling_cells = []

    for cell_id, cell in enumerate(table.cells):
        docling_cell_bbox: dict = {
            "l": cell.bbox.l / scale + table_cluster.bbox.l,
            "t": cell.bbox.t / scale + table_cluster.bbox.t,
            "r": cell.bbox.r / scale + table_cluster.bbox.l,
            "b": cell.bbox.b / scale + table_cluster.bbox.t,
            "token": "",
        }

        docling_cell: dict = {
            "cell_id": cell_id,
            "bbox": docling_cell_bbox,
            "row_span": cell.row_span,
            "col_span": cell.col_span,
            "start_row_offset_idx": cell.row,
            "end_row_offset_idx": cell.row + cell.row_span,
            "start_col_offset_idx": cell.col,
            "end_col_offset_idx": cell.col + cell.col_span,
            "indentation_level": 0,
            "text_cell_bboxes": [docling_cell_bbox],
            "column_header": False,
            "row_header": False,
            "row_section": False,
        }

        bbox = BoundingBox.model_validate(docling_cell["bbox"])

        # TODO: implement do_cell_matching
        # The image backend does not implement .get_text_in_rect()
        text_piece = page._backend.get_text_in_rect(bbox) if page._backend else ""
        docling_cell["bbox"]["token"] = text_piece

        tc = TableCell.model_validate(docling_cell)
        docling_cells.append(tc)

    docling_table = Table(
        otsl_seq=[],
        table_cells=docling_cells,
        num_rows=table.num_rows,
        num_cols=table.num_cols,
        id=table_cluster.id,
        page_no=page.page_no,
        cluster=table_cluster,
        label=table_cluster.label,
    )

    return docling_table


class CustomDoclingTableStructureOptions(BaseTableStructureOptions):
    kind: ClassVar[Literal["cells2table"]] = "cells2table"

    confidence_threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Minimum confidence score to keep a cell detection.",
        ),
    ] = Field(
        default_factory=lambda: float(os.environ.get("CELLS2TABLE_CONFIDENCE_THRESHOLD", "0.5"))
    )


class CustomDoclingTableStructureModel(BaseTableStructureModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Path | None,
        options: CustomDoclingTableStructureOptions,
        accelerator_options: AcceleratorOptions,
        **kwargs,
    ):
        self.enabled = enabled

        if self.enabled:
            if artifacts_path is None:
                models_path = None
            elif (artifacts_path / DefaultPipeline._dirname).exists():
                models_path = artifacts_path / DefaultPipeline._dirname
            else:
                models_path = artifacts_path

            self.pipeline = DefaultPipeline(models_path)

            self.options = options

            # TODO: decide how to deal with accelerator options
            # device = decide_device(accelerator_options.device)

            self.scale = 2.0  # Scale up table input images to 144 dpi

    @classmethod
    def get_options_type(cls) -> Type[BaseTableStructureOptions]:
        return CustomDoclingTableStructureOptions

    def predict_tables(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[TableStructurePrediction]:

        pages = list(pages)
        predictions: list[TableStructurePrediction] = []

        table_images: list[numpy.ndarray] = []
        table_clusters: list[Cluster] = []
        cluster_page: list[int] = []

        for i, page in enumerate(pages):
            table_prediction = page.predictions.tablestructure or TableStructurePrediction()
            page.predictions.tablestructure = table_prediction
            predictions.append(table_prediction)

            if (
                page._backend is None
                or not page._backend.is_valid()
                or page.size is None
                or page.predictions.layout is None
            ):
                continue

            clusters = [
                cluster
                for cluster in page.predictions.layout.clusters
                if cluster.label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
            ]

            if not clusters:
                continue

            cluster_page.extend([i] * len(clusters))
            table_clusters.extend(clusters)

            page_image = numpy.asarray(page.get_image(scale=self.scale))

            for cluster in clusters:
                bbox = cluster.bbox

                table_image = page_image[
                    round(bbox.t * self.scale) : round(bbox.b * self.scale),
                    round(bbox.l * self.scale) : round(bbox.r * self.scale),
                ]

                table_images.append(table_image)

        if len(table_images) == 0:
            return predictions

        with TimeRecorder(conv_res, "table_structure"):
            tables = self.pipeline(table_images, self.options.confidence_threshold)

        for table, cluster, page_idx in zip(tables, table_clusters, cluster_page):
            page = pages[page_idx]
            assert page.predictions.tablestructure is not None

            docling_table = build_docling_table(table, cluster, page, self.scale)

            page.predictions.tablestructure.table_map[cluster.id] = docling_table

            if settings.debug.visualize_tables:
                self.draw_table_and_cells(
                    conv_res,
                    page,
                    page.predictions.tablestructure.table_map.values(),
                )

        return predictions

    def draw_table_and_cells(
        self,
        conv_res: ConversionResult,
        page: Page,
        tbl_list: Iterable[Table],
        show: bool = False,
    ):
        assert page._backend is not None
        assert page.size is not None

        image = page._backend.get_page_image()  # make new image to avoid drawing on the saved ones

        scale_x = image.width / page.size.width
        scale_y = image.height / page.size.height

        draw = ImageDraw.Draw(image)

        for table_element in tbl_list:
            x0, y0, x1, y1 = table_element.cluster.bbox.as_tuple()
            y0 *= scale_y
            y1 *= scale_y
            x0 *= scale_x
            x1 *= scale_x

            draw.rectangle([(x0, y0), (x1, y1)], outline="red")

            for cell in table_element.cluster.cells:
                x0, y0, x1, y1 = cell.rect.to_bounding_box().as_tuple()
                x0 *= scale_x
                x1 *= scale_x
                y0 *= scale_y
                y1 *= scale_y

                draw.rectangle([(x0, y0), (x1, y1)], outline="green")

            for tc in table_element.table_cells:
                if tc.bbox is not None:
                    x0, y0, x1, y1 = tc.bbox.as_tuple()
                    x0 *= scale_x
                    x1 *= scale_x
                    y0 *= scale_y
                    y1 *= scale_y

                    if tc.column_header:
                        width = 3
                    else:
                        width = 1
                    draw.rectangle([(x0, y0), (x1, y1)], outline="blue", width=width)
                    draw.text(
                        (x0 + 3, y0 + 3),
                        text=f"{tc.start_row_offset_idx}, {tc.start_col_offset_idx}",
                        fill="black",
                    )
        if show:
            image.show()
        else:
            out_path: Path = (
                Path(settings.debug.debug_output_path) / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)

            out_file = out_path / f"table_struct_page_{page.page_no:05}.png"
            image.save(str(out_file), format="png")


# Plugin factory
def table_structure_engines():
    return {"table_structure_engines": [CustomDoclingTableStructureModel]}
