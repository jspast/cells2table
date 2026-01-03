"""Tests for docling plugin."""


def test_plugin_is_discoverable() -> None:
    """Test that the plugin is registered via entry points."""
    import importlib.metadata as im

    entry_points = im.entry_points(group="docling")
    names = [ep.name for ep in entry_points]
    assert "cells2table" in names, "Plugin 'cells2table' not found in entry points"


def test_model_initializes() -> None:
    """Test that CustomDoclingTableStructureModel can be imported and initialized."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    from cells2table.docling import (
        CustomDoclingTableStructureModel,
        CustomDoclingTableStructureOptions,
    )

    options = CustomDoclingTableStructureOptions()

    # Check that options instance has the 'kind' field
    assert hasattr(options, "kind"), "CustomDoclingTableStructureOptions must have a 'kind' field"
    assert options.kind == "cells2table", f"Expected kind='cells2table', got '{options.kind}'"

    accel = AcceleratorOptions()

    model = CustomDoclingTableStructureModel(
        enabled=True,
        artifacts_path=None,
        options=options,
        accelerator_options=accel,
    )
    assert model.enabled


def test_table_structure_engines_factory() -> None:
    """Test that the plugin factory returns the model."""
    from cells2table.docling import CustomDoclingTableStructureModel, table_structure_engines

    engines = table_structure_engines()
    assert "table_structure_engines" in engines
    assert len(engines["table_structure_engines"]) == 1
    assert engines["table_structure_engines"][0] is CustomDoclingTableStructureModel
