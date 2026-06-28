from argparse import Namespace

from docling_eval.datamodels.types import BenchMarkNames

from cells2table.utils.eval.main import run


def main() -> None:

    providers = ["cells2table", "TableFormer"]

    benchmarks = [
        BenchMarkNames.DOCLING_DPBENCH.value,
        BenchMarkNames.OMNIDOCBENCH.value,
        BenchMarkNames.FINTABNET.value,
        BenchMarkNames.PUBTABNET.value,
    ]

    for p in providers:
        for b in benchmarks:
            print(f"Running benchmark {b} on {p}")

            args = Namespace(
                provider=p,
                benchmark=b,
                create_gt=False,
                create_pred=True,
                evaluate=True,
                visualize=False,
                num_threads=12,
                begin=0,
                end=1000,
            )

            run(args)


if __name__ == "__main__":
    main()
