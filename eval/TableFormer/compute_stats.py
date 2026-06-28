import json
from pathlib import Path

import numpy as np
from docling_eval.datamodels.types import BenchMarkNames
from pydantic import BaseModel
from pyspark.sql import SparkSession

benchmarks_dir = Path(__file__).parents[2] / "benchmarks"


class Evaluation(BaseModel):
    TEDS: float
    filename: str
    is_complex: bool
    pred_ncols: int
    pred_nrows: int
    table_id: int
    true_ncols: int
    true_nrows: int
    timing: float


class BenchmarkStats(BaseModel):
    benchmark_name: str
    provider_name: str
    TEDS_mean: float
    TEDS_median: float
    TEDS_std: float
    timings_mean: float
    timings_std: float
    num_evaluations: int
    evaluations: list[Evaluation]


def main() -> None:

    providers = ["cells2table", "TableFormer"]

    benchmarks = [
        BenchMarkNames.DOCLING_DPBENCH.value,
        BenchMarkNames.OMNIDOCBENCH.value,
        BenchMarkNames.FINTABNET.value,
        BenchMarkNames.PUBTABNET.value,
    ]

    spark = (
        SparkSession.builder.appName("timings")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .getOrCreate()
    )

    for p in providers:
        for b in benchmarks:
            print(f"Processing stats for benchmark {b} on {p}")

            eval_json = benchmarks_dir / f"{b}/{p}/evaluations/evaluation_{b}_table_structure.json"

            parquet_dir = benchmarks_dir /  f"{b}/{p}" / ("val" if b == "PubTabNet" else "test")

            with eval_json.open() as file:
                data = json.load(file)

            df = spark.read.parquet(str(parquet_dir.resolve())).select(
                ["document_id", "prediction_timings"]
            )

            evaluations: list[Evaluation] = []
            teds: list[float] = []
            timings: list[float] = []

            for eval in data["table_structure_evaluations"]:
                if eval["structure_only_evaluation"]:
                    row = df.filter(df["document_id"] == eval["filename"]).first()
                    if row is None:
                        break

                    eval["timing"] = json.loads(row["prediction_timings"])["table_structure"][
                        eval["table_id"]
                    ]

                    timings.append(eval["timing"])
                    teds.append(eval["TEDS"])
                    evaluations.append(Evaluation.model_validate(eval))

            TEDS_mean = np.mean(teds)
            TEDS_std = np.std(teds)
            TEDS_median = np.median(teds)

            timings_mean = np.mean(timings)
            timings_std = np.std(timings)

            stats = BenchmarkStats(
                benchmark_name=b,
                provider_name=p,
                TEDS_mean=TEDS_mean,  # ty:ignore[invalid-argument-type]
                TEDS_std=TEDS_std,  # ty:ignore[invalid-argument-type]
                TEDS_median=TEDS_median,  # ty:ignore[invalid-argument-type]
                timings_mean=timings_mean,  # ty:ignore[invalid-argument-type]
                timings_std=timings_std,  # ty:ignore[invalid-argument-type]
                num_evaluations=len(evaluations),
                evaluations=evaluations,
            )

            stats_json = stats.model_dump_json(indent=2)

            with open(Path(__file__).parent / f"data/{b}_{p}.json", "w") as file:
                file.write(stats_json)


if __name__ == "__main__":
    main()
