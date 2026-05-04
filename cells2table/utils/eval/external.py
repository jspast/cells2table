import gc
from io import BytesIO
from pathlib import Path

import numpy as np
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import (
    DocumentConverter,
    FormatOption,
    ImageFormatOption,
    PdfFormatOption,
)
from docling.experimental.datamodel.table_crops_layout_options import TableCropsLayoutOptions
from docling_core.types.io import DocumentStream
from paddleocr import TableRecognitionPipelineV2
from rapidocr import OCRVersion

from cells2table.docling import CustomDoclingTableStructureOptions

benchmarks_dir = Path(__file__).parents[3] / "benchmarks"


def cells2table_pdfpipelineoptions(num_threads: int) -> PdfPipelineOptions:
    options = PdfPipelineOptions()

    options.allow_external_plugins = True
    options.do_table_structure = True
    options.layout_options = TableCropsLayoutOptions(keep_empty_clusters=True)
    options.table_structure_options = CustomDoclingTableStructureOptions()
    options.images_scale = 2.0

    options.do_ocr = True
    options.ocr_options = RapidOcrOptions(
        force_full_page_ocr=True,
        rapidocr_params={
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        },
    )

    options.accelerator_options = AcceleratorOptions(num_threads=num_threads)

    options.generate_page_images = True  # Needed for visualizations

    return options


def cells2table_formatoptions(num_threads: int) -> dict[InputFormat, FormatOption]:
    return {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=cells2table_pdfpipelineoptions(num_threads)
        ),
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_options=cells2table_pdfpipelineoptions(num_threads)
        ),
    }


def create_gt(ds: Dataset, benchmark_dir: Path):
    gt_dir = benchmark_dir / "gt"
    gt_dir.mkdir(exist_ok=True)

    for sample in ds:
        with open(gt_dir / f"{sample['sample_id']}.html", "w") as f:
            f.write(sample["ground_truth_html"])


def create_pred_cells2table(ds: Dataset, benchmark_dir: Path):
    pred_dir = benchmark_dir / "cells2table"
    pred_dir.mkdir(exist_ok=True)

    converter = DocumentConverter(format_options=cells2table_formatoptions(12))

    for sample in ds:
        with open(pred_dir / f"{sample['sample_id']}.html", "w") as f:
            pil_image = sample["image"]
            buf = BytesIO()
            pil_image.save(buf, format="PNG")
            buf.seek(0)

            stream = DocumentStream(name="image.png", stream=buf)

            result = converter.convert(stream)
            f.write(result.document.tables[0].export_to_html(result.document))

            if result.input and result.input._backend:
                result.input._backend.unload()
            del result
            buf.close()
            pil_image.close()
            gc.collect()


def create_pred_pp(ds: Dataset, benchmark_dir: Path):
    pred_dir = benchmark_dir / "pp"
    pred_dir.mkdir(exist_ok=True)

    pipeline = TableRecognitionPipelineV2(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        engine="transformers",
    )

    for sample in ds:
        with open(pred_dir / f"{sample['sample_id']}.html", "w") as f:
            pil_image = sample["image"]
            image = np.asarray(pil_image)

            output = pipeline.predict(image)

            print(output[0].html)
            f.write(output[0].html["table_1"])

            pil_image.close()
            gc.collect()


def main():
    ds = load_dataset("pulse-ai/PulseBench-Tab", split="train[534:902]")

    benchmark_dir = benchmarks_dir / "PulseBench-Tab"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    # create_gt(ds, benchmark_dir)
    # create_pred_cells2table(ds, benchmark_dir)
    create_pred_pp(ds, benchmark_dir)

    # To score, run:
    # uv run cells2table/utils/eval/tlag_scorer.py --gt benchmarks/PulseBench-Tab/gt/ --pred benchmarks/PulseBench-Tab/cells2table/ --output benchmarks/PulseBench-Tab/cells2table_scores.json


if __name__ == "__main__":
    main()
