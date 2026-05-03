import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import numpy as np
from numpy.typing import NDArray

from cells2table.pipelines import DefaultPipeline
from cells2table.utils.eval.cells2table_provider import Cells2tablePredictionProvider
from cells2table.utils.visualize import bgr_to_rgb, rgb_to_bgr, visualize_detections

try:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
    from docling.datamodel.settings import settings
    from docling.document_converter import FormatOption, ImageFormatOption, PdfFormatOption
    from docling_eval.cli import main as eval_main
    from docling_eval.datamodels.types import BenchMarkNames, EvaluationModality
    from docling_eval.dataset_builders.dataset_builder import BaseEvaluationDatasetBuilder
    from docling_eval.dataset_builders.doclingdpbench_builder import DoclingDPBenchDatasetBuilder
    from docling_eval.dataset_builders.omnidocbench_builder import OmniDocBenchDatasetBuilder
    from docling_eval.prediction_providers.base_prediction_provider import BasePredictionProvider
    from docling_eval.prediction_providers.docling_provider import DoclingPredictionProvider
    from docling_eval.prediction_providers.tableformer_provider import TableFormerPredictionProvider
    from docling_eval.utils.external_predictions_visualizer import PredictionsVisualizer
    from PIL import Image

    from cells2table.docling import CustomDoclingTableStructureOptions
except ImportError:
    raise ImportError("docling-eval is not installed. Unable to initialize evaluation.")

benchmarks_dir = Path(__file__).parents[3] / "benchmarks"


def pil_to_cv2(image: Image.Image) -> NDArray[np.uint8]:
    return rgb_to_bgr(np.array(image))


def cv2_to_pil(image: NDArray[np.uint8]) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(image))


def analyze_image(
    image: NDArray[np.uint8],
    detection_model_idx: int | None = None,
) -> NDArray[np.uint8]:
    table_pipeline = DefaultPipeline()
    table, detections = table_pipeline.debug(image, detection_model_idx)
    return visualize_detections(image, detections)  # ty:ignore[invalid-return-type]


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

    def create_gt(self, begin: int = 0, end: int = -1) -> None:
        dataset = self.builder(target=self.base_dir / "gt", begin_index=begin, end_index=end)  # type: ignore
        dataset.retrieve_input_dataset()
        dataset.save_to_disk()

    def create_pred(
        self,
        provider: BasePredictionProvider,
        dirname: str,
        begin: int = 0,
        end: int = -1,
    ) -> None:
        provider.create_prediction_dataset(
            name=self.name,
            gt_dataset_dir=self.base_dir / "gt",
            target_dataset_dir=self.base_dir / dirname,
            begin_index=begin,
            end_index=end,
        )

    def evaluate(self, modality: EvaluationModality, dirname: str) -> None:
        eval_main.evaluate(
            modality=modality,
            benchmark=self.name,
            idir=self.base_dir / dirname,
            odir=self.base_dir / dirname / "evaluations",
        )

        eval_main.visualize(
            modality=modality,
            benchmark=self.name,
            idir=self.base_dir / dirname,
            odir=self.base_dir / dirname / "evaluations",
        )

    def visualize(self, dirname: str) -> None:
        visualizer = PredictionsVisualizer(self.base_dir / dirname / "visualizations")
        visualizer.create_visualizations(dataset_dir=self.base_dir / dirname)


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
    parser.add_argument("provider", type=str, default="cells2table")
    parser.add_argument("-b", "--benchmark", type=str, default="DoclingDPBench")
    parser.add_argument("-g", "--create-gt", action="store_true", default=False)
    parser.add_argument("-p", "--create-pred", action="store_true", default=False)
    parser.add_argument("-e", "--evaluate", action="store_true", default=False)
    parser.add_argument("-v", "--visualize", action="store_true", default=False)
    parser.add_argument("-t", "--num-threads", type=int, default=4)
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    settings.debug.profile_pipeline_timings = True

    provider = None
    match args.provider.lower():
        case "cells2table":
            provider = Cells2tablePredictionProvider(
                num_threads=args.num_threads,
                do_visualization=True,
            )
        case "tableformer":
            provider = TableFormerPredictionProvider(
                num_threads=args.num_threads,
                do_visualization=True,
            )

    if provider is None:
        raise ValueError(f'Unrecognized provider "{args.provider}". Unable to initialize.')

    benchmark = None
    for b in [DoclingDPBench(), OmniDocBench()]:
        if b.name.lower() == args.benchmark.lower():
            benchmark = b
            break
    if benchmark is None:
        raise ValueError(f'Unrecognized benchmark "{args.benchmark}". Unable to initialize.')

    if args.create_gt:
        benchmark.create_gt(args.begin, args.end)

    if args.create_pred:
        benchmark.create_pred(provider, args.provider, args.begin, args.end)

    if args.evaluate:
        benchmark.evaluate(EvaluationModality.TABLE_STRUCTURE, args.provider)

    if args.visualize:
        benchmark.visualize(args.provider)


if __name__ == "__main__":
    main()
