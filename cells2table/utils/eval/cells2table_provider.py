import copy
import logging
from importlib.metadata import version
from pathlib import Path
from typing import override

import numpy
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus
from docling.utils.profiling import ProfilingItem, TimeRecorder
from docling_core.types.doc.document import DoclingDocument, TableCell, TableData, TableItem
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.io import DocumentStream
from docling_eval.datamodels.dataset_record import DatasetRecord, DatasetRecordWithPrediction
from docling_eval.prediction_providers.base_prediction_provider import BasePredictionProvider
from docling_eval.prediction_providers.tableformer_provider import (
    TableFormerPredictionProvider,
    TableFormerUpdater,
)
from docling_eval.utils.utils import insert_images_from_pil
from PIL import Image

from cells2table.docling import (
    CustomDoclingTableStructureModel,
    CustomDoclingTableStructureOptions,
)

_log = logging.getLogger(__name__)


class Cells2tablePredictionProvider(TableFormerPredictionProvider):
    @override
    def __init__(
        self,
        num_threads: int = 12,
        artifacts_path: Path | None = None,
        do_visualization: bool = False,
        ignore_missing_predictions: bool = True,
        true_labels: set[DocItemLabel] | None = None,
        pred_labels: set[DocItemLabel] | None = None,
    ):
        """
        Initialize the cells2table prediction provider.

        Args:
            num_threads: Number of threads for prediction
            artifacts_path: Path to artifacts
            do_visualization: Whether to generate visualizations
            ignore_missing_predictions: Whether to ignore missing predictions
            true_labels: Set of DocItemLabel to use for ground truth visualization
            pred_labels: Set of DocItemLabel to use for prediction visualization
        """
        BasePredictionProvider.__init__(
            self,
            do_visualization=do_visualization,
            ignore_missing_predictions=ignore_missing_predictions,
            true_labels=true_labels,
            pred_labels=pred_labels,
        )
        self.tf_updater: Cells2tableUpdater = Cells2tableUpdater(num_threads, artifacts_path)

    @override
    def predict(self, record: DatasetRecord) -> DatasetRecordWithPrediction:
        """
        Generate a prediction for table structure.

        Args:
            record: Input dataset record

        Returns:
            Dataset record with prediction

        Raises:
            RuntimeError: If ground truth doc is not available or if mime type is unsupported
        """
        if record.ground_truth_doc is None:
            raise RuntimeError(
                "true_doc must be given for cells2table prediction provider to work."
            )

        updated = False
        pred_doc = None
        timings = {}

        try:
            if record.mime_type == "application/pdf":
                if not isinstance(record.original, DocumentStream):
                    raise RuntimeError("Original document must be a DocumentStream for PDF files")

                # Process PDF
                updated, pred_doc, timings = self.tf_updater.replace_tabledata(
                    copy.deepcopy(record.original.stream), record.ground_truth_doc
                )

            elif record.mime_type == "image/png":
                # Process image
                updated, pred_doc, timings = self.tf_updater.replace_tabledata_with_image(
                    record.ground_truth_doc,
                    record.ground_truth_page_images,
                )
            else:
                raise RuntimeError(
                    f"Unsupported mime type: {record.mime_type}. cells2table supports 'application/pdf' and 'application/png'"
                )

            pred_doc = insert_images_from_pil(
                pred_doc,
                record.ground_truth_pictures,
                record.ground_truth_page_images,
            )
            # Set status based on update success
            status = ConversionStatus.SUCCESS if updated else ConversionStatus.FAILURE

        except Exception as e:
            _log.error(f"Error in cells2table prediction: {str(e)}")
            status = ConversionStatus.FAILURE
            if not self.ignore_missing_predictions:
                raise
            pred_doc = record.ground_truth_doc.model_copy(
                deep=True
            )  # Use copy of ground truth as fallback

        pred_record = self.create_dataset_record_with_prediction(record, pred_doc, timings=timings)
        pred_record.status = status
        return pred_record

    def info(self) -> dict:
        """Get information about the prediction provider."""
        return {
            "asset": "cells2table",
            "version": version("cells2table"),
        }


class Cells2tableUpdater(TableFormerUpdater):
    """
    Utility class for updating table data using cells2table.

    This class handles the prediction of table structures using the cell2table.
    """

    @override
    def __init__(
        self,
        num_threads: int = 12,
        artifacts_path: Path | None = None,
    ):
        """
        Initialize the cells2table updater.

        Args:
            num_threads: Number of threads for prediction
            artifacts_path: Path to artifacts
        """
        table_structure_options = CustomDoclingTableStructureOptions()
        accelerator_options = AcceleratorOptions(
            num_threads=num_threads, device=AcceleratorDevice.AUTO
        )
        self._docling_tf_model: CustomDoclingTableStructureModel = CustomDoclingTableStructureModel(
            enabled=True,
            artifacts_path=artifacts_path,
            options=table_structure_options,
            accelerator_options=accelerator_options,
        )
        self._docling_tf_model.scale = 1.0
        _log.info("Initialized cells2table")

    def replace_tabledata_with_image(
        self,
        true_doc: DoclingDocument,
        true_page_images: list[Image.Image],
    ) -> tuple[bool, DoclingDocument, dict[str, ProfilingItem]]:
        pred_doc = copy.deepcopy(true_doc)
        updated = False
        timings: dict[str, ProfilingItem] = {}

        class _TimingContainer:
            def __init__(self, timings: dict[str, ProfilingItem]):
                self.timings = timings

        timing_container = _TimingContainer(timings)

        # Ensure document has exactly one page
        if len(pred_doc.pages) != 1:
            _log.error("Document must have exactly one page")
            return False, pred_doc, timings

        page_size = pred_doc.pages[1].size
        page_image = numpy.array(true_page_images[0])

        # Process each table item
        for item, level in pred_doc.iterate_items():
            if isinstance(item, TableItem):
                for prov in item.prov:
                    try:
                        bbox = prov.bbox.to_top_left_origin(page_image.shape[0])

                        # Ensure bounding box is within page bounds
                        bbox.l = max(bbox.l, 0.0)
                        bbox.t = max(bbox.t, 0.0)
                        bbox.r = min(bbox.r, page_size.width)
                        bbox.b = min(bbox.b, page_size.height)

                        table_image = page_image[
                            round(bbox.t) : round(bbox.b),
                            round(bbox.l) : round(bbox.r),
                        ]

                        with TimeRecorder(timing_container, "table_structure"):  # ty:ignore[invalid-argument-type]
                            tables = self._docling_tf_model.pipeline(
                                [table_image], self._docling_tf_model.options.confidence_threshold
                            )

                        table = tables[0]

                        docling_cells = []

                        for cell_id, cell in enumerate(table.cells):
                            docling_cell_bbox: dict = {
                                "l": cell.bbox.l + bbox.l,
                                "t": cell.bbox.t + bbox.t,
                                "r": cell.bbox.r + bbox.l,
                                "b": cell.bbox.b + bbox.t,
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

                            tc = TableCell.model_validate(docling_cell)
                            docling_cells.append(tc)

                        table_data: TableData = TableData(
                            num_rows=table.num_rows,
                            num_cols=table.num_cols,
                            table_cells=docling_cells,
                        )

                        # Update item data
                        item.data = table_data
                        updated = True

                    except Exception as e:
                        _log.error(f"Error predicting table: {str(e)}")
                        raise

        return updated, pred_doc, timings
