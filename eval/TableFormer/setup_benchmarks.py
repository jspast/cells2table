from argparse import Namespace
from pathlib import Path

from docling_eval.datamodels.types import BenchMarkNames

from cells2table.utils.eval.main import run

benchmarks_dir = Path(__file__).parents[2] / "benchmarks"


def main() -> None:

    benchmarks = [
        BenchMarkNames.DOCLING_DPBENCH.value,
        BenchMarkNames.OMNIDOCBENCH.value,
        BenchMarkNames.FINTABNET.value,
        BenchMarkNames.PUBTABNET.value,
    ]

    for b in benchmarks:
        if (benchmarks_dir / f"{b}/gt").exists():
            print(f"Benchmark {b} is already downloaded")
            continue

        print(f"Downloading benchmark {b}")

        args = Namespace(
            provider="cells2table",
            benchmark=b,
            create_gt=True,
            create_pred=False,
            evaluate=False,
            visualize=False,
            num_threads=12,
            begin=0,
            end=1000,
        )

        run(args)


if __name__ == "__main__":
    main()
