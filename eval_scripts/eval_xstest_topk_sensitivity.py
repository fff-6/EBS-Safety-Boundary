"""Run XSTest Top-K sensitivity analysis for EBS.

This script keeps the model, memory, router, prompts, decoding parameters,
judge, and evaluation criteria fixed while varying only the number of
retrieved experience rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import tiktoken

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from eval_scripts.eval_ebs_main_experiment import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXPERIENCE_TOKEN_BUDGET,
    DEFAULT_TARGET_MODEL,
    MAIN_BENIGN_BENCHMARK,
    MAIN_HARMFUL_BENCHMARK,
    MAIN_ATTACK_METHODS,
    ModelConfig,
    RedEvalRunConfig,
    _build_model_config,
    build_main_metrics,
    load_redeval_default_config,
    resolve_api_key,
    run_redeval_redbench,
    save_main_metric_outputs,
)
from ebs.core.experience_bank import format_experiences_for_prompt, select_experiences  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join("data", "ebs", "eval", "xstest_topk_sensitivity")
DEFAULT_K_VALUES = (0, 4, 8, 12, 16)


def _load_project_env() -> None:
    """Load repository-level `.env` files into the current process if present."""

    def _parse_env_line(line: str) -> tuple[str, str] | None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            return None
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            return None
        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return key, value

    for env_name in (".env.full", ".env"):
        env_path = ROOT_DIR / env_name
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def _parse_k_values(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("K values cannot be empty.")
    result = [int(item) for item in items]
    if any(k < 0 for k in result):
        raise ValueError("K values must be non-negative.")
    return result


def _parse_benchmark_limits(value: str | None) -> dict[str, int] | None:
    if value is None:
        return None
    limits: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid benchmark limit `{item}`. Expected `Benchmark:Count`.")
        benchmark, count = item.split(":", 1)
        limits[benchmark.strip()] = int(count.strip())
    return limits or None


def _build_target_model(args: argparse.Namespace, default_mapping: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        provider=args.target_provider or default_mapping.get("provider"),
        model=args.target_model or default_mapping.get("model") or DEFAULT_TARGET_MODEL,
        api_key=args.target_api_key if args.target_api_key is not None else resolve_api_key(default_mapping),
        base_url=args.target_base_url if args.target_base_url is not None else default_mapping.get("base_url"),
        temperature=args.target_temperature
        if "--target_temperature" in sys.argv
        else float(default_mapping.get("temperature", 0.6)),
        top_p=args.target_top_p if "--target_top_p" in sys.argv else float(default_mapping.get("top_p", 0.9)),
        max_tokens=args.target_max_tokens
        if "--target_max_tokens" in sys.argv
        else int(default_mapping.get("max_tokens", args.target_max_tokens)),
        max_model_len=args.target_max_model_len
        if "--target_max_model_len" in sys.argv
        else (
            int(default_mapping["max_model_len"]) if default_mapping.get("max_model_len") is not None else None
        ),
        gpu_memory_utilization=args.target_gpu_memory_utilization
        if "--target_gpu_memory_utilization" in sys.argv
        else (
            float(default_mapping["gpu_memory_utilization"])
            if default_mapping.get("gpu_memory_utilization") is not None
            else None
        ),
        request_interval_seconds=float(default_mapping.get("request_interval_seconds", 0.0)),
    )


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _find_eval_results(run_dir: Path, split_name: str, model_output_name: str) -> Path:
    path = run_dir / "logs" / "xstest_official" / split_name / "XSTest" / "base" / model_output_name / "eval_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval results for split={split_name}: {path}")
    return path


def _build_selected_rule_metadata(
    *,
    experience_bank: dict[str, dict[str, str]],
    query: str,
    harmful_label: int,
    k_value: int,
    token_budget: int,
    tokenizer: tiktoken.Encoding,
) -> dict[str, Any]:
    if k_value <= 0:
        return {
            "selected_bucket": None,
            "retrieved_rule_ids": [],
            "retrieved_rule_text": "None",
            "num_exp_tokens": 0,
        }

    selected_bucket, selected = select_experiences(
        experience_bank,
        problem=query,
        harmful_label=harmful_label,
        max_experiences=k_value,
        token_budget=token_budget,
    )
    formatted = format_experiences_for_prompt(selected)
    num_tokens = 0 if formatted == "None" else len(tokenizer.encode(formatted))
    return {
        "selected_bucket": selected_bucket,
        "retrieved_rule_ids": list(selected.keys()),
        "retrieved_rule_text": formatted,
        "num_exp_tokens": num_tokens,
    }


def _extract_label(judges: list[str]) -> str:
    return judges[0].strip() if judges else ""


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XSTest Top-K sensitivity analysis for EBS.")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_PATH), help="RedEval config path.")
    parser.add_argument("--experience_file", type=str, required=True, help="Experience bank used by EBS.")
    parser.add_argument(
        "--experiment_prefix",
        type=str,
        default="qwen3_8b_xstest_topk_sensitivity",
        help="Prefix for per-K experiment folders and output files.",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="0,4,8,12,16",
        help="Comma-separated K values to evaluate.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join("dataset", "redbench_dataset.json"),
        help="Path to the local RedBench dataset JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to store per-K runs and aggregated CSV files.",
    )
    parser.add_argument(
        "--experience_token_budget",
        type=int,
        default=DEFAULT_EXPERIENCE_TOKEN_BUDGET,
        help="Approximate token budget for retrieved experiences. Use 0 to disable truncation.",
    )
    parser.add_argument(
        "--benchmark_limits",
        type=str,
        default=None,
        help="Optional debug limits, e.g. `XSTestSafe:2,XSTestUnsafe:2`.",
    )
    parser.add_argument("--target_provider", type=str, default=None, choices=["openai", "vllm", "local_transformers"])
    parser.add_argument("--target_model", type=str, default=None, help="Target/base model override.")
    parser.add_argument("--target_base_url", type=str, default=None, help="Target model base URL override.")
    parser.add_argument("--target_api_key", type=str, default=None, help="Target model API key override.")
    parser.add_argument("--target_temperature", type=float, default=0.6, help="Target model temperature.")
    parser.add_argument("--target_top_p", type=float, default=0.9, help="Target model top-p.")
    parser.add_argument("--target_max_tokens", type=int, default=256, help="Target model max output tokens.")
    parser.add_argument("--target_max_model_len", type=int, default=4096, help="vLLM max_model_len.")
    parser.add_argument("--target_gpu_memory_utilization", type=float, default=0.8, help="vLLM GPU utilization.")
    parser.add_argument("--judge_provider", type=str, default=None, choices=["openai", "vllm", "local_transformers"])
    parser.add_argument("--judge_model", type=str, default=None, help="Override both attack/refuse judge models.")
    parser.add_argument("--judge_base_url", type=str, default=None, help="Judge model base URL override.")
    parser.add_argument("--judge_api_key", type=str, default=None, help="Judge model API key override.")
    parser.add_argument("--judge_temperature", type=float, default=0.0, help="Judge model temperature.")
    parser.add_argument("--judge_top_p", type=float, default=0.9, help="Judge model top-p.")
    parser.add_argument("--judge_max_tokens", type=int, default=16, help="Judge model max output tokens.")
    parser.add_argument("--judge_max_model_len", type=int, default=4096, help="Judge vLLM max_model_len.")
    parser.add_argument("--judge_gpu_memory_utilization", type=float, default=0.8, help="Judge vLLM GPU utilization.")
    return parser.parse_args()


def main() -> None:
    _load_project_env()
    args = parse_args()
    k_values = _parse_k_values(args.k_values)
    benchmark_limits = _parse_benchmark_limits(args.benchmark_limits)
    default_config = load_redeval_default_config(args.config_path)
    default_target_mapping = default_config.get("target", {})
    default_judge_mapping = default_config.get("judge", {})
    default_attack_judge_mapping = default_config.get("attack_judge", default_judge_mapping)
    default_refuse_judge_mapping = default_config.get("refuse_judge", default_judge_mapping)

    target_model = _build_target_model(args, default_target_mapping)
    if not target_model.provider:
        raise ValueError(f"Target provider is not configured. Update {args.config_path} or pass --target_provider.")

    judge_model = _build_model_config(
        default_mapping=default_judge_mapping,
        provider=args.judge_provider,
        model=args.judge_model,
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        temperature=args.judge_temperature,
        temperature_flag="--judge_temperature",
        top_p=args.judge_top_p,
        top_p_flag="--judge_top_p",
        max_tokens=args.judge_max_tokens,
        max_tokens_flag="--judge_max_tokens",
        max_tokens_override=None,
        max_model_len=args.judge_max_model_len,
        max_model_len_flag="--judge_max_model_len",
        gpu_memory_utilization=args.judge_gpu_memory_utilization,
        gpu_memory_flag="--judge_gpu_memory_utilization",
    )
    attack_judge_model = _build_model_config(
        default_mapping=default_attack_judge_mapping or default_judge_mapping,
        provider=args.judge_provider,
        model=args.judge_model,
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        temperature=args.judge_temperature,
        temperature_flag="--judge_temperature",
        top_p=args.judge_top_p,
        top_p_flag="--judge_top_p",
        max_tokens=args.judge_max_tokens,
        max_tokens_flag="--judge_max_tokens",
        max_tokens_override=None,
        max_model_len=args.judge_max_model_len,
        max_model_len_flag="--judge_max_model_len",
        gpu_memory_utilization=args.judge_gpu_memory_utilization,
        gpu_memory_flag="--judge_gpu_memory_utilization",
    )
    refuse_judge_model = _build_model_config(
        default_mapping=default_refuse_judge_mapping or default_judge_mapping,
        provider=args.judge_provider,
        model=args.judge_model,
        api_key=args.judge_api_key,
        base_url=args.judge_base_url,
        temperature=args.judge_temperature,
        temperature_flag="--judge_temperature",
        top_p=args.judge_top_p,
        top_p_flag="--judge_top_p",
        max_tokens=args.judge_max_tokens,
        max_tokens_flag="--judge_max_tokens",
        max_tokens_override=None,
        max_model_len=args.judge_max_model_len,
        max_model_len_flag="--judge_max_model_len",
        gpu_memory_utilization=args.judge_gpu_memory_utilization,
        gpu_memory_flag="--judge_gpu_memory_utilization",
    )
    if judge_model is None or attack_judge_model is None or refuse_judge_model is None:
        raise ValueError(f"Judge model is not configured. Update {args.config_path} or pass judge overrides.")

    experience_bank = _load_json(Path(args.experience_file))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    output_dir = Path(args.output_dir)
    aggregate_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []

    for k_value in k_values:
        experiment_name = f"{args.experiment_prefix}_k{k_value}"
        print(f"[topk] running K={k_value} -> experiment_name={experiment_name}")
        summary = run_redeval_redbench(
            RedEvalRunConfig(
                experiment_name=experiment_name,
                dataset_path=args.dataset_path,
                output_dir=str(output_dir),
                target_model=target_model,
                judge_model=judge_model,
                attack_judge_model=attack_judge_model,
                refuse_judge_model=refuse_judge_model,
                benchmarks={MAIN_BENIGN_BENCHMARK},
                harmful_benchmarks={MAIN_HARMFUL_BENCHMARK},
                benign_benchmarks={MAIN_BENIGN_BENCHMARK},
                benchmark_limits=benchmark_limits,
                dataset_truncate=None,
                experience_file=args.experience_file,
                experience_top_k=k_value,
                experience_token_budget=args.experience_token_budget,
                attack_methods=MAIN_ATTACK_METHODS,
                human_jailbreak_subsets=1,
                attack_max_tokens=None,
                refuse_max_tokens=None,
                xstest_official=True,
            )
        )
        save_main_metric_outputs(summary, str(output_dir), experiment_name)
        main_metrics = build_main_metrics(summary)
        run_dir = output_dir / experiment_name
        safe_eval = _load_json(_find_eval_results(run_dir, "safe", target_model.output_name))
        unsafe_eval = _load_json(_find_eval_results(run_dir, "unsafe_contrast", target_model.output_name))
        split_points = [("safe", safe_eval), ("unsafe_contrast", unsafe_eval)]

        exp_token_values: list[float] = []
        latency_values: list[float] = []
        for subset_name, points in split_points:
            for point in points:
                metadata = _build_selected_rule_metadata(
                    experience_bank=experience_bank,
                    query=str(point["query"]),
                    harmful_label=int(point["harmful_label"]),
                    k_value=k_value,
                    token_budget=args.experience_token_budget,
                    tokenizer=tokenizer,
                )
                latency = float(point.get("inference_latency_seconds", 0.0))
                exp_token_values.append(float(metadata["num_exp_tokens"]))
                latency_values.append(latency)
                per_sample_rows.append(
                    {
                        "sample_id": point["sample_id"],
                        "subset": subset_name,
                        "K": k_value,
                        "input": point["query"],
                        "retrieved_rule_ids": "|".join(metadata["retrieved_rule_ids"]),
                        "num_exp_tokens": metadata["num_exp_tokens"],
                        "response": point.get("responses", [""])[0] if point.get("responses") else "",
                        "latency": latency,
                        "judge_label": _extract_label(point.get("judges", [])),
                    }
                )

        aggregate_rows.append(
            {
                "K": k_value,
                "avg_exp_tokens": round(mean(exp_token_values), 4) if exp_token_values else 0.0,
                "latency_mean": round(mean(latency_values), 4) if latency_values else 0.0,
                "latency_std": round(pstdev(latency_values), 4) if len(latency_values) > 1 else 0.0,
                "ASR": round(float(main_metrics["xstest_unsafe_asr"]) * 100.0, 4),
                "BRR": round(float(main_metrics["xstest_safe_over_refusal"]) * 100.0, 4),
                "Pair_Accuracy": round(float(main_metrics["xstest_pair_accuracy"]) * 100.0, 4),
            }
        )

    aggregate_csv_path = output_dir / f"{args.experiment_prefix}_summary.csv"
    per_sample_csv_path = output_dir / f"{args.experiment_prefix}_per_sample.csv"
    _write_csv(
        aggregate_csv_path,
        aggregate_rows,
        ["K", "avg_exp_tokens", "latency_mean", "latency_std", "ASR", "BRR", "Pair_Accuracy"],
    )
    _write_csv(
        per_sample_csv_path,
        per_sample_rows,
        ["sample_id", "subset", "K", "input", "retrieved_rule_ids", "num_exp_tokens", "response", "latency", "judge_label"],
    )
    print(f"[topk] saved aggregate CSV to {aggregate_csv_path}")
    print(f"[topk] saved per-sample CSV to {per_sample_csv_path}")


if __name__ == "__main__":
    main()
