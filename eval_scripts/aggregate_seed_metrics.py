"""Aggregate experiment metrics from independent seeded runs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, pstdev

METRICS = ("XSTest BRR", "XSTest ASR", "XSTest Pair Accuracy", "HarmBench ASR")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.input_glob))
    if len(paths) != 5:
        raise ValueError(f"Five-seed aggregation requires exactly five runs; found {len(paths)}")
    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    aggregate: dict[str, object] = {"num_seeds": 5, "source_files": paths, "metrics": {}}
    metrics = aggregate["metrics"]
    assert isinstance(metrics, dict)
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if values:
            metrics[metric] = {"mean": mean(values), "std": pstdev(values), "values": values}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
