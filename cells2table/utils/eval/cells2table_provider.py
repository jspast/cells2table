import copy
import logging
from importlib.metadata import version
from pathlib import Path
from typing import override

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.io import DocumentStream
from docling_eval.datamodels.dataset_record import DatasetRecord, DatasetRecordWithPrediction
from docling_eval.prediction_providers.base_prediction_provider import BasePredictionProvider
from docling_eval.prediction_providers.tableformer_provider import (
    TableFormerPredictionProvider,
    TableFormerUpdater,
)
from docling_eval.utils.utils import insert_images_from_pil

from cells2table.docling import CustomDoclingTableStructureModel, CustomDoclingTableStructureOptions

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
        self.tf_updater = Cells2tableUpdater(num_threads, artifacts_path)

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

        try:
            if record.mime_type == "application/pdf":
                if not isinstance(record.original, DocumentStream):
                    raise RuntimeError("Original document must be a DocumentStream for PDF files")

                # Process PDF
                updated, pred_doc = self.tf_updater.replace_tabledata(
                    copy.deepcopy(record.original.stream), record.ground_truth_doc
                )

            # elif record.mime_type == "image/png":
            #     # Process image
            #     updated, pred_doc = self.tf_updater.replace_tabledata_with_page_tokens(
            #         record.ground_truth_doc,
            #         record.ground_truth_page_images,
            #     )
            else:
                raise RuntimeError(
                    f"Unsupported mime type: {record.mime_type}. cells2table supports only 'application/pdf' for the moment"
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

        pred_record = self.create_dataset_record_with_prediction(record, pred_doc, None)
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
        self._docling_tf_model = CustomDoclingTableStructureModel(
            enabled=True,
            artifacts_path=artifacts_path,
            options=table_structure_options,
            accelerator_options=accelerator_options,
        )
        _log.info("Initialized cells2table")
