# cells2table

Parsing tables in document images with cell detection models

## Implemented pipelines

### PaddlePaddle

- Classification model (wired / wireless)
- Cell detection model with different weights for each class

## Inference runtimes

Currently OpenCV, ONNX Runtime and Transformers are supported. OpenCV is the default.

## Instalation

With [uv](https://docs.astral.sh/uv/), add to your project using:

```sh
uv add cells2table
```

### Extras

| Optional       | Description                       |
|----------------|-----------------------------------|
| `huggingface`  | For downloading models from HF    |
| `onnxruntime`  | For using ONNX Runtime as runtime |
| `transformers` | For using transformers as runtime |

## Usage

cells2table only extracts structural information from the tables. Another library is needed to extract content from the cells.

### Quick visual demo

```sh
cells2table path/to/image.png
```

### Docling (recommended)

A [docling plugin](https://docling-project.github.io/docling/concepts/plugins/) is provided to allow integrating cells2table in a complete pipeline.

Usage example:

```python
from cells2table.docling import CustomDoclingTableStructureOptions

pipeline_options = PdfPipelineOptions(
    allow_external_plugins=True,
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
