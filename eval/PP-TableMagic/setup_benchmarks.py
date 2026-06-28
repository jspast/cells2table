from pathlib import Path

from datasets.load import load_dataset

from cells2table.utils.eval.external import create_gt

benchmarks_dir = Path(__file__).parents[2] / "benchmarks"


def main() -> None:

    benchmark_dir = benchmarks_dir / "PulseBench-Tab"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    ch_split = "train[225:385]"
    en_split = "train[534:902]"
    splits = [ch_split, en_split]

    for split in splits:
        ds = load_dataset("pulse-ai/PulseBench-Tab", split=split)
        create_gt(ds, benchmark_dir)


if __name__ == "__main__":
    main()
