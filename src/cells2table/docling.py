import copy
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal, Optional, Sequence, Type

import numpy
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
from docling_core.types.doc.page import BoundingRectangle, TextCellUnit
from PIL import ImageDraw

# from docling.utils.accelerator_utils import decide_device
from .models import DefaultPipeline


def get_tokens(page: Page, table_cluster: Cluster, scale: float) -> list[str]:
    # Check if word-level cells are available from backend:
    sp = page._backend.get_segmented_page() if page._backend else None
    if sp is not None:
        tcells = sp.get_cells_in_bbox(
            cell_unit=TextCellUnit.WORD,
            bbox=table_cluster.bbox,
        )
        if len(tcells) == 0:
            # In case word-level cells yield empty
            tcells = table_cluster.cells
    else:
        # Otherwise - we use normal (line/phrase) cells
        tcells = table_cluster.cells
    tokens = []
    for c in tcells:
        # Only allow non empty strings (spaces) into the cells of a table
        if len(c.text.strip()) > 0:
            new_cell = copy.deepcopy(c)
            new_cell.rect = BoundingRectangle.from_bounding_box(
                new_cell.rect.to_bounding_box().scaled(scale=scale)
            )
            tokens.append(
                {
                    "id": new_cell.index,
                    "text": new_cell.text,
                    "bbox": new_cell.rect.to_bounding_box().model_dump(),
                }
            )

    return tokens


class CustomDoclingTableStructureOptions(BaseTableStructureOptions):
    kind: ClassVar[Literal["cells2table"]] = "cells2table"


class CustomDoclingTableStructureModel(BaseTableStructureModel):
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: CustomDoclingTableStructureOptions,
        accelerator_options: AcceleratorOptions,
    ):
        self.enabled = enabled

        if self.enabled:
            self.pipeline = DefaultPipeline(artifacts_path)

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

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                existing_prediction = page.predictions.tablestructure or TableStructurePrediction()
                page.predictions.tablestructure = existing_prediction
                predictions.append(existing_prediction)
                continue

            with TimeRecorder(conv_res, "table_structure"):
                assert page.predictions.layout is not None
                assert page.size is not None

                table_prediction = TableStructurePrediction()
                page.predictions.tablestructure = table_prediction

                in_tables = [
                    (
                        cluster,
                        [
                            round(cluster.bbox.l) * self.scale,
                            round(cluster.bbox.t) * self.scale,
                            round(cluster.bbox.r) * self.scale,
                            round(cluster.bbox.b) * self.scale,
                        ],
                    )
                    for cluster in page.predictions.layout.clusters
                    if cluster.label in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
                ]
                if not in_tables:
                    predictions.append(table_prediction)
                    continue

                page_input: dict = {
                    "width": page.size.width * self.scale,
                    "height": page.size.height * self.scale,
                    "image": numpy.asarray(page.get_image(scale=self.scale)),
                }

                for table_cluster, tbl_box in in_tables:
                    page_input["tokens"] = get_tokens(page, table_cluster, self.scale)

                    table_image = page_input["image"][
                        round(tbl_box[1]) : round(tbl_box[3]),
                        round(tbl_box[0]) : round(tbl_box[2]),
                    ]

                    table = self.pipeline([table_image])[0]

                    docling_cells = []

                    for cell_id, cell in enumerate(table.cells):
                        docling_cell_bbox: dict = {
                            "l": (cell.bbox.l + tbl_box[0]) / self.scale,
                            "t": (cell.bbox.t + tbl_box[1]) / self.scale,
                            "r": (cell.bbox.r + tbl_box[0]) / self.scale,
                            "b": (cell.bbox.b + tbl_box[1]) / self.scale,
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

                    page.predictions.tablestructure.table_map[table_cluster.id] = docling_table

                if settings.debug.visualize_tables:
                    self.draw_table_and_cells(
                        conv_res,
                        page,
                        page.predictions.tablestructure.table_map.values(),
                    )

                predictions.append(table_prediction)

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
