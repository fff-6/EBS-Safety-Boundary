"""Validate the local files required by the EBS full experiment protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROMPT_FIELDS = ("prompt", "problem", "question", "instruction", "query")
CONSTRUCTION_FILES = {
    "harmful": (Path("dataset/wildjailbreak_harmful.jsonl"), 400),
    "benign": (Path("dataset/star_benign.jsonl"), 200),
    "ethics": (Path("dataset/evil_data.jsonl"), 200),
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not any(str(row.get(field, "")).strip() for field in PROMPT_FIELDS) for row in rows):
        raise ValueError(f"Every row must contain one prompt field: {path}")
    return rows


def prompt_text(row: dict) -> str:
    for field in PROMPT_FIELDS:
        value = str(row.get(field, "")).strip()
        if value:
            return value
    return ""


def main() -> None:
    construction_prompts: set[str] = set()
    selection_manifest: dict[str, object] = {"selection_policy": "first_n_after_local_preparation", "buckets": {}}
    buckets = selection_manifest["buckets"]
    assert isinstance(buckets, dict)
    for bucket, (path, minimum) in CONSTRUCTION_FILES.items():
        rows = read_jsonl(path)
        if len(rows) < minimum:
            raise ValueError(f"{bucket} requires at least {minimum} rows; found {len(rows)} in {path}")
        selected = rows[:minimum]
        prompts = [prompt_text(row) for row in selected]
        if len(prompts) != len(set(prompts)):
            raise ValueError(f"Duplicate prompts in the selected {bucket} construction slice")
        construction_prompts.update(prompts)
        buckets[bucket] = {
            "source": path.as_posix(),
            "selected_indices": list(range(minimum)),
            "prompt_sha256": [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts],
        }
        print(f"[ok] {bucket}: {len(rows)} available, first {minimum} selected")

    redbench_path = Path("dataset/redbench_dataset.json")
    if not redbench_path.exists():
        raise FileNotFoundError(redbench_path)
    redbench = json.loads(redbench_path.read_text(encoding="utf-8"))
    for benchmark in ("HarmBench", "XSTest"):
        if benchmark not in redbench:
            raise ValueError(f"Missing {benchmark} in {redbench_path}")

    serialized = json.dumps(redbench, ensure_ascii=False)
    exact_overlaps = [prompt for prompt in construction_prompts if prompt and prompt in serialized]
    if exact_overlaps:
        raise ValueError(f"Found {len(exact_overlaps)} exact construction/evaluation prompt overlaps")
    manifest_path = Path("artifacts/data_indices/full_construction_selection.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(selection_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[ok] HarmBench and XSTest are present; no exact construction/evaluation overlap found")
    print(f"[ok] wrote selected indices and prompt hashes to {manifest_path}")
    print("Experiment input validation passed")


if __name__ == "__main__":
    main()
