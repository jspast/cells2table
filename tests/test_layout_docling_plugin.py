"""Tests for layout Docling plugin."""

import importlib.metadata as im
from pathlib import Path

import pytest
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PipelineOptions
from docling.document_converter import (
    DocumentConverter,
    ImageFormatOption,
    PdfFormatOption,
)

from cells2table.docling import (
    CustomDoclingLayoutModel,
    CustomDoclingLayoutOptions,
)
from cells2table.docling.layout import layout_engines

from .gt_utils import verify_text


def test_plugin_is_discoverable() -> None:
    """Test that the plugin is registered via entry points."""

    entry_points = im.entry_points(group="docling")
    names = [ep.name for ep in entry_points]
    assert "ppdoclayoutv3" in names, "Plugin 'ppdoclayoutv3' not found in entry points"


def test_model_initializes() -> None:
    """Test that CustomDoclingTableStructureModel can be imported and initialized."""

    options = CustomDoclingLayoutOptions()

    # Check that options instance has the 'kind' field
    assert hasattr(options, "kind"), "CustomDoclingLayoutOptions must have a 'kind' field"
    assert options.kind == "ppdoclayoutv3", f"Expected kind='ppdoclayoutv3', got '{options.kind}'"

    model = CustomDoclingLayoutModel(
        artifacts_path=None,
        options=options,
        accelerator_options=AcceleratorOptions(),
    )
    assert model.options == options


def test_table_structure_engines_factory() -> None:
    """Test that the plugin factory returns the model."""

    engines = layout_engines()
    assert "layout_engines" in engines
    assert len(engines["layout_engines"]) == 1
    assert engines["layout_engines"][0] is CustomDoclingLayoutModel


@pytest.fixture
def pipeline_options() -> PipelineOptions:
    return PdfPipelineOptions(
        allow_external_plugins=True,
        layout_options=CustomDoclingLayoutOptions(),
        do_ocr=False,
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
    return Path(__file__).parent / "data" / "images" / "layout.pdf"


@pytest.fixture
def gt_file_path() -> Path:
    return Path(__file__).parent / "data" / "gt" / "layout.md"


def test_conversion(converter: DocumentConverter, test_file_path: Path, gt_file_path: Path) -> None:
    result = converter.convert(test_file_path)
    md = result.document.export_to_markdown()

    verify_text(gt_file_path, md)
