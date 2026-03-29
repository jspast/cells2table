import argparse
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from cells2table.docling import CustomDoclingTableStructureOptions
from cells2table.pipelines import DefaultPipeline
from cells2table.utils.visualize import create_visualization

try:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    from docling.document_converter import FormatOption, ImageFormatOption, PdfFormatOption
    from docling_eval.cli import main as eval_main
    from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
    from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
    from docling_eval.dataset_builders.doclingdpbench_builder import DoclingDPBenchDatasetBuilder
    from docling_eval.dataset_builders.omnidocbench_builder import OmniDocBenchDatasetBuilder
    from docling_eval.prediction_providers.base_prediction_provider import BasePredictionProvider
    from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider
    from docling_eval.utils.external_predictions_visualizer import PredictionsVisualizer
except ImportError:
    raise ImportError("docling-eval is not installed. Unable to initialize evaluation.")

benchmarks_dir = Path(__file__).parent.parent.parent / "benchmarks"


def pil_to_cv2(image: Image.Image) -> NDArray[np.uint8]:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # ty:ignore[invalid-return-type]


def cv2_to_pil(image: NDArray[np.uint8]) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def analyze_image(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    table_pipeline = DefaultPipeline()
    tables = table_pipeline([image])
    return create_visualization(image, tables[0])


def cells2table_pdfpipelineoptions(num_threads: int) -> PdfPipelineOptions:
    options = PdfPipelineOptions()

    options.allow_external_plugins = True
    options.do_table_structure = True
    options.table_structure_options = CustomDoclingTableStructureOptions()
    options.images_scale = 2.0

    options.ocr_options = RapidOcrOptions(backend="torch")

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


def cells2table_provider(num_threads: int) -> DoclingPredictionProvider:
    provider = DoclingPredictionProvider(
        format_options=cells2table_formatoptions(num_threads),
        do_visualization=True,
    )
    provider.prediction_modalities = [EvaluationModality.TABLE_STRUCTURE]

    return provider


class BaseDataset(ABC):
    """Base interface for datasets."""

    @property
    @abstractmethod
    def name(self) -> BenchMarkNames:
        pass

    @property
    @abstractmethod
    def builder(self) -> type[BaseEvaluationDatasetBuilder]:
        pass

    @property
    def base_dir(self) -> Path:
        return benchmarks_dir / self.name

    def create_gt(self) -> None:
        dataset = self.builder(target=self.base_dir / "gt")  # type: ignore
        dataset.retrieve_input_dataset()
        dataset.save_to_disk()

    def create_pred(self, provider: BasePredictionProvider) -> None:
        provider.create_prediction_dataset(
            name=self.name,
            gt_dataset_dir=self.base_dir / "gt",
            target_dataset_dir=self.base_dir / "pred",
        )

    def evaluate(self, modality: EvaluationModality) -> None:
        eval_main.evaluate(
            modality=modality,
            benchmark=self.name,
            idir=self.base_dir / "pred",
            odir=self.base_dir / "evaluations" / modality.value,
        )

        eval_main.visualize(
            modality=modality,
            benchmark=self.name,
            idir=self.base_dir / "pred",
            odir=self.base_dir / "evaluations" / modality.value,
        )

    def visualize(self) -> None:
        visualizer = PredictionsVisualizer(self.base_dir / "pred" / "visualizations")
        visualizer.create_visualizations(dataset_dir=self.base_dir / "pred")


class OmniDocBench(BaseDataset):
    @property
    def name(self) -> BenchMarkNames:
        return BenchMarkNames.OMNIDOCBENCH

    @property
    def builder(self) -> type[BaseEvaluationDatasetBuilder]:
        return OmniDocBenchDatasetBuilder


class DoclingDPBench(BaseDataset):
    @property
    def name(self) -> BenchMarkNames:
        return BenchMarkNames.DOCLING_DPBENCH

    @property
    def builder(self) -> type[BaseEvaluationDatasetBuilder]:
        return DoclingDPBenchDatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="cells2table")
    parser.add_argument("--benchmark", type=str, default="DoclingDPBench")
    parser.add_argument("--create-gt", action="store_true", default=False)
    parser.add_argument("--create-pred", action="store_true", default=False)
    parser.add_argument("--evaluate", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    provider = cells2table_provider(args.num_threads)

    benchmark = None
    for b in [DoclingDPBench(), OmniDocBench()]:
        if b.name.lower() == args.benchmark.lower():
            benchmark = b
            break
    if benchmark is None:
        raise ValueError(f'Unrecognized benchmark "{args.benchmark}". Unable to initialize.')

    if args.create_gt:
        benchmark.create_gt()

    if args.create_pred:
        benchmark.create_pred(provider)

    if args.evaluate:
        benchmark.evaluate(EvaluationModality.TABLE_STRUCTURE)

    if args.visualize:
        benchmark.visualize()


if __name__ == "__main__":
    main()
