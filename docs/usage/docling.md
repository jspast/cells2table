# Docling

[Docling](https://docling-project.github.io/docling/) is a great Python library for parsing documents of diverse formats.
It features a [plugins system](https://docling-project.github.io/docling/concepts/plugins/) that allows extending its capabilities with custom models.

Using Docling is the recommended way to integrate our pipelines and models in a complete pipeline, with native text extraction from PDFs, OCR, picture annotation, serialization to formats such as Markdown, and hierarchical chunking.

## CLI

Starting from Docling [v2.120.0](https://github.com/docling-project/docling/releases/tag/v2.120.0), table structure and layout plugins can be used directly with `docling` CLI.

To check the available external plugins, one can run:

``` sh
docling --show-external-plugins
```

If cells2table is properly installed, it should show some plugins:

```
           Available layout engines
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃          Name ┃ Plugin        ┃ Package     ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ ppdoclayoutv3 │ ppdoclayoutv3 │ cells2table │
└───────────────┴───────────────┴─────────────┘
          Available table engines
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃        Name ┃ Plugin      ┃ Package     ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ cells2table │ cells2table │ cells2table │
└─────────────┴─────────────┴─────────────┘
```

Those plugins can then be used easily:

``` sh
docling --allow-external-plugins --table-structure-engine=cells2table --layout-engine=ppdoclayoutv3 <source>
```

## Python

Using from Python is also easy:

``` python hl_lines="7-9"
from cells2table.docling import (
    CustomDoclingLayoutOptions,
    CustomDoclingTableStructureOptions,
)

pipeline_options = PdfPipelineOptions(
    allow_external_plugins=True,
    layout_options=CustomDoclingLayoutOptions(),
    table_structure_options=CustomDoclingTableStructureOptions(),
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
    }
)

result = converter.convert("path/to/document.pdf")
print(result.document.export_to_markdown())
```
