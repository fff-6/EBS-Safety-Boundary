import json
import os
from typing import Any

from ebs.core.experience_bank import CATEGORY_KEYS


def _get_data_dir() -> str:
    return os.environ.get("EBS_DATA_DIR", "dataset")


def _ensure_exists(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {name} at: {path}")


def _require(module: str, install_hint: str):
    try:
        return __import__(module)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Missing dependency `{module}`. Install {install_hint}.") from e


def _parse_recipe(name: str) -> list[tuple[str, int | None]]:
    if os.path.exists(name):
        return [(name, None)]

    recipe = []
    for part in name.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            dataset_name, count = part.split(":", 1)
            recipe.append((dataset_name.strip(), int(count.strip())))
        else:
            recipe.append((part, None))
    if not recipe:
        raise ValueError("Dataset recipe is empty.")
    return recipe


def _infer_experience_bucket(record: dict[str, Any], dataset_name: str, harmful_label: int) -> str:
    explicit = record.get("experience_bucket")
    if explicit in CATEGORY_KEYS:
        return str(explicit)

    dataset_lower = dataset_name.lower()
    if "evil" in dataset_lower or "ethic" in dataset_lower or "moral" in dataset_lower:
        return "ethics"
    return "harmful" if int(harmful_label) else "benign"


def _normalize_record(record: dict[str, Any], dataset_name: str, default_harmful_label: int) -> dict[str, Any]:
    prompt = (
        record.get("prompt")
        or record.get("problem")
        or record.get("question")
        or record.get("instruction")
        or record.get("query")
    )
    if prompt is None and "jailbreak instruction" in record:
        prompt = record["jailbreak instruction"]
    if prompt is None:
        raise ValueError(f"Cannot find prompt field for dataset `{dataset_name}`: {record.keys()}")

    groundtruth = record.get("groundtruth")
    if isinstance(groundtruth, dict):
        reference_response = (
            record.get("response")
            or record.get("reference_response")
            or record.get("target_response")
            or record.get("answer")
            or groundtruth.get("reference_response")
            or ""
        )
        harmful_label = int(record.get("harmful_label", groundtruth.get("harmful_label", default_harmful_label)))
        level = int(record.get("level", groundtruth.get("level", 1)))
        dataset_meta = groundtruth.get("dataset", dataset_name)
    else:
        reference_response = (
            record.get("response")
            or record.get("reference_response")
            or record.get("target_response")
            or record.get("answer")
            or ""
        )
        harmful_label = int(record.get("harmful_label", default_harmful_label))
        level = int(record.get("level", 1))
        dataset_meta = dataset_name

    experience_bucket = _infer_experience_bucket(record, dataset_name=dataset_name, harmful_label=harmful_label)

    return {
        "problem": prompt,
        "groundtruth": {
            "harmful_label": harmful_label,
            "level": level,
            "reference_response": reference_response,
            "dataset": dataset_meta,
            "experience_bucket": experience_bucket,
        },
    }


def _load_from_local_file(path: str) -> list[dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        data = json.load(open(path, encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", [])
        return data
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if ext == ".csv":
        pd = _require("pandas", "`pip install pandas`")
        return pd.read_csv(path).to_dict("records")
    raise ValueError(f"Unsupported local dataset file type: {path}")


def _load_builtin_dataset(dataset_name: str, max_item: int | None = None) -> list[dict[str, Any]]:
    data_dir = _get_data_dir()

    if dataset_name == "stratasword":
        pd = _require("pandas", "`pip install pandas`")
        records = []
        for level in (1, 2, 3):
            csv_path = os.path.join(data_dir, f"strata_sword_en_level_{level}.csv")
            _ensure_exists(csv_path, f"stratasword level-{level} csv")
            df = pd.read_csv(csv_path)
            df["level"] = level
            df = df.rename(columns={"jailbreak instruction": "prompt"})
            df["harmful_label"] = 1
            records.extend(df.to_dict("records"))
        return records[:max_item] if max_item is not None else records

    if dataset_name == "sorrybench":
        pd = _require("pandas", "`pip install pandas`")
        csv_path = os.path.join(data_dir, "sorrybench.csv")
        _ensure_exists(csv_path, "sorrybench csv")
        df = pd.read_csv(csv_path)
        if "prompt_style" in df.columns:
            allowed = [
                "base",
                "authority_endorsement",
                "evidence-based_persuasion",
                "expert_endorsement",
                "logical_appeal",
                "role_play",
            ]
            df = df[df["prompt_style"].isin(allowed)]
        df["harmful_label"] = 1
        records = df.to_dict("records")
        return records[:max_item] if max_item is not None else records

    if dataset_name in {"jailbreakbench", "wildjailbreak", "strongreject", "xstest", "starbenign", "star", "harmbench"}:
        _require("pandas", "`pip install pandas`")
        load_dataset = _require("datasets", "`pip install datasets`").load_dataset

        if dataset_name == "jailbreakbench":
            path = os.path.join(data_dir, "JailbreakBench")
            _ensure_exists(path, "JailbreakBench dataset folder")
            df = load_dataset(path)["train"].to_pandas()
            if "subset" in df.columns:
                df = df[df["subset"] == "harmful"]
            df["harmful_label"] = 1
        elif dataset_name == "wildjailbreak":
            path = os.path.join(data_dir, "wildjailbreak")
            _ensure_exists(path, "wildjailbreak dataset folder")
            df = load_dataset(path, "eval", delimiter="\t", keep_default_na=False)["train"].to_pandas()
            if "label" in df.columns:
                df = df[df["label"] == 1]
            if "adversarial" in df.columns:
                df["prompt"] = df["adversarial"]
            df["harmful_label"] = 1
        elif dataset_name == "strongreject":
            path = os.path.join(data_dir, "StrongReject")
            _ensure_exists(path, "StrongReject dataset folder")
            df = load_dataset(path)["train"].to_pandas()
            df["harmful_label"] = 1
        elif dataset_name == "xstest":
            path = os.path.join(data_dir, "XsTest")
            _ensure_exists(path, "XsTest dataset folder")
            df = load_dataset(path)["test"].to_pandas()
            if "label" in df.columns:
                df = df[df["label"] == "safe"]
            df["harmful_label"] = 0
        elif dataset_name == "starbenign":
            path = os.path.join(data_dir, "STAR-benign")
            _ensure_exists(path, "STAR-benign dataset folder")
            df = load_dataset(path)["train"].to_pandas()
            if "question" in df.columns:
                df["prompt"] = df["question"]
            df["harmful_label"] = 0
        elif dataset_name == "star":
            path = os.path.join(data_dir, "STAR-1K")
            _ensure_exists(path, "STAR-1K dataset folder")
            df = load_dataset(path)["train"].to_pandas()
            if "question" in df.columns:
                df["prompt"] = df["question"]
            df["harmful_label"] = 1
        else:
            path = os.path.join(data_dir, "Harmbench")
            _ensure_exists(path, "Harmbench dataset folder")
            df = load_dataset(path, "standard")["train"].to_pandas()
            df["harmful_label"] = 1

        records = df.to_dict("records")
        return records[:max_item] if max_item is not None else records

    raise ValueError(f"Unsupported EBS dataset: {dataset_name}")


def load_data(name: str) -> list[dict[str, Any]]:
    recipe = _parse_recipe(name)
    merged = []
    for dataset_name, count in recipe:
        if os.path.exists(dataset_name):
            rows = _load_from_local_file(dataset_name)
            for row in rows[:count] if count is not None else rows:
                merged.append(
                    _normalize_record(row, dataset_name=os.path.basename(dataset_name), default_harmful_label=1)
                )
            continue

        rows = _load_builtin_dataset(dataset_name, max_item=count)
        default_label = 0 if dataset_name in {"xstest", "starbenign"} else 1
        merged.extend(
            [_normalize_record(row, dataset_name=dataset_name, default_harmful_label=default_label) for row in rows]
        )
    return merged
