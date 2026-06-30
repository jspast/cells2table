import csv
import json
from argparse import Namespace
from pathlib import Path

from pydantic import BaseModel

from cells2table.utils.eval.rd_scorer import run as run_rd
from cells2table.utils.eval.teds_scorer import run as run_teds
from cells2table.utils.eval.tlag_scorer import run as run_tlag

benchmarks_dir = Path(__file__).parents[2] / "benchmarks"


class Evaluation(BaseModel):
    filename: str
    TEDS: float
    TLAG: float
    rd: float
    timing: float


class Metric(BaseModel):
    mean: float
    median: float
    std: float
    p10: float
    p25: float
    p75: float
    p90: float
    perfect_count: int


class BenchmarkStats(BaseModel):
    benchmark_name: str
    provider_name: str
    total_time: float
    TEDS: Metric
    TLAG: Metric
    rd: Metric
    num_evaluations: int
    evaluations: list[Evaluation]


def main() -> None:
    providers = ["cells2table", "pp"]
    benchmarks = ["PulseBench-Tab"]

    for p in providers:
        for b in benchmarks:
            b_dir = benchmarks_dir / b

            teds_file = b_dir / f"{p}_teds.json"
            tlag_file = b_dir / f"{p}_tlag.json"
            rd_file = b_dir / f"{p}_rd.json"
            timings_file = b_dir / f"{p}_timings.csv"

            teds_args = Namespace(
                gt=b_dir / "gt",
                pred=b_dir / p,
                output=teds_file,
                workers=8,
            )
            run_teds(teds_args)

            tlag_args = Namespace(
                gt=b_dir / "gt",
                pred=b_dir / p,
                output=tlag_file,
                workers=8,
            )
            run_tlag(tlag_args)

            rd_args = Namespace(
                gt=b_dir / "gt",
                pred=b_dir / p,
                output=rd_file,
                workers=8,
            )
            run_rd(rd_args)

            with open(teds_file) as file:
                teds_data = json.load(file)

            with open(tlag_file) as file:
                tlag_data = json.load(file)

            with open(rd_file) as file:
                rd_data = json.load(file)

            total_time = 0

            with open(timings_file) as file:
                timings_csv = csv.reader(file)
                next(timings_csv)  # skip header

                evaluations: list[Evaluation] = []

                for teds, tlag, rd, timing in zip(
                    teds_data["per_sample"].values(),
                    tlag_data["per_sample"].values(),
                    rd_data["per_sample"].values(),
                    timings_csv,
                ):
                    evaluations.append(
                        Evaluation(
                            filename=timing[0],
                            TEDS=teds,
                            TLAG=tlag,
                            rd=rd,
                            timing=float(timing[1]),
                        )
                    )

                    total_time += float(timing[1])

            def build_metric(data: dict) -> Metric:
                return Metric(
                    mean=data["summary"]["mean"],
                    median=data["summary"]["median"],
                    std=data["summary"]["std"],
                    p10=data["summary"]["p10"],
                    p25=data["summary"]["p25"],
                    p75=data["summary"]["p75"],
                    p90=data["summary"]["p90"],
                    perfect_count=data["summary"]["perfect_count"],
                )

            stats = BenchmarkStats(
                benchmark_name=b,
                provider_name=p,
                total_time=total_time,
                TEDS=build_metric(teds_data),
                TLAG=build_metric(tlag_data),
                rd=build_metric(rd_data),
                num_evaluations=len(evaluations),
                evaluations=evaluations,
            )

            stats_json = stats.model_dump_json(indent=2)

            with open(Path(__file__).parent / f"data/{b}_{p}.json", "w") as file:
                file.write(stats_json)


if __name__ == "__main__":
    main()
