"""Download official RedBench subsets and write the JSON layout used by EBS."""

import argparse
import json
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/redbench_dataset.json")
    args = parser.parse_args()

    repository = "knoveleng/redbench"
    output: dict[str, dict[str, list[dict]]] = {}
    for config_name in get_dataset_config_names(repository):
        dataset = load_dataset(repository, config_name)
        output[config_name] = {
            split_name: [dict(record) for record in split]
            for split_name, split in dataset.items()
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False)
    print(f"Saved {len(output)} RedBench subsets to {output_path}")


if __name__ == "__main__":
    main()
