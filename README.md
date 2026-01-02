# cells2table

Parsing tables in document images with cell detection models

## Implemented pipelines

### PaddlePaddle

- Classification model (wired / wireless)
- Cell detection model with different weights for each class

Uses ONNX weights downloaded automatically from [Hugging Face](https://huggingface.co/jspast/paddlepaddle-table-models-onnx) on first use.

## Instalation

With [uv](https://docs.astral.sh/uv/), add to your project with:

```sh
uv add cells2table
```

ONNX models need a [ONNX Runtime](https://onnxruntime.ai/getting-started) installed to run. You can install one on your own or use one of the optionals already configured.

| Optional        | Description             |
| --------------- | ----------------------- |
| `docling`       | For docling usage       |
| `huggingface`   | For downloading models  |
| `onnx_cuda`     | For NVIDIA GPUs         |
| `onnx_openvino` | For Intel GPUs and CPUs |
| `onnx_cpu`      | Default CPU runtime     |

## Usage

cells2table only extract structural information from the tables. Another library is needed to extract content from the cells.

### Docling

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
        InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
    }
)

result = converter.convert("path/to/document.pdf")
print(result.document.export_to_markdown())
```
