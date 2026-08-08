"""Tests for docling plugin."""

import importlib.metadata as im
from pathlib import Path

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PipelineOptions, RapidOcrOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

from cells2table.docling import (
    CustomDoclingTableStructureModel,
    CustomDoclingTableStructureOptions,
)
from cells2table.docling.table import table_structure_engines

from .gt_utils import verify_text


def test_plugin_is_discoverable() -> None:
    """Test that the plugin is registered via entry points."""

    entry_points = im.entry_points(group="docling")
    names = [ep.name for ep in entry_points]
    assert "cells2table" in names, "Plugin 'cells2table' not found in entry points"


def test_model_initializes() -> None:
    """Test that CustomDoclingTableStructureModel can be imported and initialized."""

    options = CustomDoclingTableStructureOptions()

    # Check that options instance has the 'kind' field
    assert hasattr(options, "kind"), "CustomDoclingTableStructureOptions must have a 'kind' field"
    assert options.kind == "cells2table", f"Expected kind='cells2table', got '{options.kind}'"

    model = CustomDoclingTableStructureModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    assert model.enabled


def test_table_structure_engines_factory() -> None:
    """Test that the plugin factory returns the model."""

    engines = table_structure_engines()
    assert "table_structure_engines" in engines
    assert len(engines["table_structure_engines"]) == 1
    assert engines["table_structure_engines"][0] is CustomDoclingTableStructureModel


@pytest.fixture
def pipeline_options() -> PipelineOptions:
    return PdfPipelineOptions(
        allow_external_plugins=True,
        table_structure_options=CustomDoclingTableStructureOptions(),
        do_ocr=True,
        ocr_options=RapidOcrOptions(),
        generate_page_images=True,
    )


@pytest.fixture
def converter(pipeline_options: PipelineOptions) -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        },
    )


@pytest.fixture
def test_file_path() -> Path:
    return Path(__file__).parent / "data" / "images" / "wired.png"


@pytest.fixture
def gt_file_path() -> Path:
    return Path(__file__).parent / "data" / "gt" / "wired.md"


def test_conversion(converter: DocumentConverter, test_file_path: Path, gt_file_path: Path) -> None:
    result = converter.convert(test_file_path)
    md = result.document.export_to_markdown()

    # An error here can be caused by an OCR change
    verify_text(gt_file_path, md)
