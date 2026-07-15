import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Allow `python eval_scripts/eval_ebs_redbench.py` from the repository root.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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


_load_project_env()


def _load_redeval_helpers():
    module_path = ROOT_DIR / "ebs" / "runtime" / "eval" / "redeval_redbench.py"
    module_name = "redeval_redbench_lib"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_HELPERS = _load_redeval_helpers()
DEFAULT_ATTACK_METHODS = _HELPERS.DEFAULT_ATTACK_METHODS
DEFAULT_BENIGN_BENCHMARKS = _HELPERS.DEFAULT_BENIGN_BENCHMARKS
DEFAULT_HARMFUL_BENCHMARKS = _HELPERS.DEFAULT_HARMFUL_BENCHMARKS
DEFAULT_CONFIG_PATH = _HELPERS.DEFAULT_CONFIG_PATH
ModelConfig = _HELPERS.ModelConfig
RedEvalRunConfig = _HELPERS.RedEvalRunConfig
_parse_benchmark_limits = _HELPERS._parse_benchmark_limits
_parse_csv_set = _HELPERS._parse_csv_set
build_model_config_from_mapping = _HELPERS.build_model_config_from_mapping
load_redeval_default_config = _HELPERS.load_redeval_default_config
resolve_api_key = _HELPERS._resolve_api_key
run_redeval_redbench = _HELPERS.run_redeval_redbench
DEFAULT_EXPERIENCE_TOP_K = _HELPERS.DEFAULT_EXPERIENCE_TOP_K
DEFAULT_EXPERIENCE_TOKEN_BUDGET = _HELPERS.DEFAULT_EXPERIENCE_TOKEN_BUDGET


def _build_model_config(
    *,
    default_mapping: dict,
    provider: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
    temperature: float,
    temperature_flag: str,
    top_p: float,
    top_p_flag: str,
    max_tokens: int,
    max_tokens_flag: str,
    max_tokens_override: int | None,
    max_model_len: int,
    max_model_len_flag: str,
    gpu_memory_utilization: float,
    gpu_memory_flag: str,
):
    if not (provider or model or default_mapping):
        return None
    return ModelConfig(
        provider=provider or default_mapping.get("provider"),
        model=model or default_mapping.get("model"),
        api_key=api_key if api_key is not None else resolve_api_key(default_mapping),
        base_url=base_url if base_url is not None else default_mapping.get("base_url"),
        temperature=temperature
        if temperature_flag in sys.argv
        else float(default_mapping.get("temperature", 0.6)),
        top_p=top_p if top_p_flag in sys.argv else float(default_mapping.get("top_p", 0.9)),
        max_tokens=max_tokens
        if max_tokens_flag in sys.argv
        else (max_tokens_override if max_tokens_override is not None else int(default_mapping.get("max_tokens", 16))),
        max_model_len=max_model_len
        if max_model_len_flag in sys.argv
        else (int(default_mapping["max_model_len"]) if default_mapping.get("max_model_len") is not None else None),
        gpu_memory_utilization=gpu_memory_utilization
        if gpu_memory_flag in sys.argv
        else (
            float(default_mapping["gpu_memory_utilization"])
            if default_mapping.get("gpu_memory_utilization") is not None
            else None
        ),
        request_interval_seconds=float(default_mapping.get("request_interval_seconds", 0.0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EBS on RedBench with the official RedEval workflow.")
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Repository-level RedEval config file path.",
    )
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name used for output files.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=os.path.join("dataset", "redbench_dataset.json"),
        help="Path to the local RedBench dataset JSON file.",
    )
    parser.add_argument(
        "--experience_file",
        type=str,
        default=None,
        help="Optional JSON file containing generated EBS experience texts.",
    )
    parser.add_argument(
        "--experience_top_k",
        type=int,
        default=DEFAULT_EXPERIENCE_TOP_K,
        help="Maximum number of retrieved experiences injected into the EBS prompt.",
    )
    parser.add_argument(
        "--experience_token_budget",
        type=int,
        default=DEFAULT_EXPERIENCE_TOKEN_BUDGET,
        help="Approximate token budget for retrieved experiences injected into the EBS prompt. Use 0 to disable truncation.",
    )
    parser.add_argument(
        "--dataset_truncate",
        type=int,
        default=None,
        help="Debug only: truncate the flattened dataset to the first N samples after official filtering.",
    )
    parser.add_argument(
        "--benchmark_limits",
        type=str,
        default=None,
        help="Debug only: per-benchmark sample limits after official dataset selection.",
    )
    parser.add_argument(
        "--attack_methods",
        type=str,
        default=None,
        help="Comma-separated attack methods override, e.g. `direct,human_jailbreak`.",
    )
    parser.add_argument(
        "--human_jailbreak_subsets",
        type=int,
        default=None,
        help="Number of human jailbreak templates to use per harmful sample; -1 means all.",
    )
    parser.add_argument(
        "--speed_profile",
        type=str,
        default="full",
        choices=["full", "fair_fast", "smoke"],
        help="Speed preset. `full` preserves current behavior; `fair_fast` keeps benchmark logic but lowers token budgets; `smoke` is for quick pipeline checks.",
    )
    parser.add_argument("--target_provider", type=str, default=None, choices=["openai", "vllm", "local_transformers"])
    parser.add_argument("--target_model", type=str, default=None, help="Target model name.")
    parser.add_argument("--target_base_url", type=str, default=None, help="Target model base URL.")
    parser.add_argument("--target_api_key", type=str, default=None, help="Target model API key.")
    parser.add_argument("--target_temperature", type=float, default=0.6, help="Target model temperature.")
    parser.add_argument("--target_top_p", type=float, default=0.9, help="Target model top-p.")
    parser.add_argument("--target_max_tokens", type=int, default=1024, help="Target model max output tokens.")
    parser.add_argument("--target_max_model_len", type=int, default=4096, help="vLLM max_model_len.")
    parser.add_argument(
        "--target_gpu_memory_utilization",
        type=float,
        default=0.8,
        help="vLLM gpu_memory_utilization.",
    )
    parser.add_argument("--judge_provider", type=str, default=None, choices=["openai", "vllm", "local_transformers"])
    parser.add_argument("--judge_model", type=str, default=None, help="Judge model name.")
    parser.add_argument("--judge_base_url", type=str, default=None, help="Judge model base URL.")
    parser.add_argument("--judge_api_key", type=str, default=None, help="Judge model API key.")
    parser.add_argument("--judge_temperature", type=float, default=0.6, help="Judge model temperature.")
    parser.add_argument("--judge_top_p", type=float, default=0.9, help="Judge model top-p.")
    parser.add_argument("--judge_max_tokens", type=int, default=16, help="Judge model max output tokens.")
    parser.add_argument("--judge_max_model_len", type=int, default=4096, help="Judge vLLM max_model_len.")
    parser.add_argument(
        "--judge_gpu_memory_utilization",
        type=float,
        default=0.8,
        help="Judge vLLM gpu_memory_utilization.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("data", "ebs", "eval"),
        help="Directory used to store RedEval logs and summary files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_config = load_redeval_default_config(args.config_path)
    default_target_mapping = default_config.get("target", {})
    default_judge_mapping = default_config.get("judge", {})
    default_attack_judge_mapping = default_config.get("attack_judge", default_judge_mapping)
    default_refuse_judge_mapping = default_config.get("refuse_judge", default_judge_mapping)
    default_defaults = default_config.get("defaults", {})
    harmful_benchmarks_cfg = default_defaults.get("harmful_benchmarks")
    benign_benchmarks_cfg = default_defaults.get("benign_benchmarks")
    attack_methods_cfg = default_defaults.get("attack_methods")

    harmful_benchmarks = (
        DEFAULT_HARMFUL_BENCHMARKS if harmful_benchmarks_cfg is None else set(harmful_benchmarks_cfg)
    )
    benign_benchmarks = DEFAULT_BENIGN_BENCHMARKS if benign_benchmarks_cfg is None else set(benign_benchmarks_cfg)
    attack_methods = DEFAULT_ATTACK_METHODS if attack_methods_cfg is None else tuple(attack_methods_cfg)
    human_jailbreak_subsets = int(default_defaults.get("human_jailbreak_subsets", 1))
    if args.attack_methods:
        attack_methods = tuple(method.strip() for method in args.attack_methods.split(",") if method.strip())
    if args.human_jailbreak_subsets is not None:
        human_jailbreak_subsets = args.human_jailbreak_subsets

    attack_max_tokens_override: int | None = None
    refuse_max_tokens_override: int | None = None
    judge_max_tokens_override: int | None = None
    if args.speed_profile == "fair_fast":
        attack_methods = tuple(method for method in attack_methods if method in {"direct"}) or ("direct",)
        attack_max_tokens_override = 192
        refuse_max_tokens_override = 128
        judge_max_tokens_override = 16
        human_jailbreak_subsets = min(human_jailbreak_subsets, 1) if human_jailbreak_subsets != -1 else 1
    elif args.speed_profile == "smoke":
        attack_methods = ("direct",)
        attack_max_tokens_override = 96
        refuse_max_tokens_override = 64
        judge_max_tokens_override = 16
        human_jailbreak_subsets = 1

    target_model = ModelConfig(
        provider=args.target_provider or default_target_mapping.get("provider"),
        model=args.target_model or default_target_mapping.get("model"),
        api_key=args.target_api_key if args.target_api_key is not None else default_target_mapping.get("api_key"),
        base_url=args.target_base_url if args.target_base_url is not None else default_target_mapping.get("base_url"),
        temperature=(
            args.target_temperature
            if "--target_temperature" in sys.argv
            else float(default_target_mapping.get("temperature", 0.6))
        ),
        top_p=args.target_top_p if "--target_top_p" in sys.argv else float(default_target_mapping.get("top_p", 0.9)),
        max_tokens=(
            args.target_max_tokens if "--target_max_tokens" in sys.argv else int(default_target_mapping.get("max_tokens", 1024))
        ),
        max_model_len=(
            args.target_max_model_len
            if "--target_max_model_len" in sys.argv
            else (
                int(default_target_mapping["max_model_len"])
                if default_target_mapping.get("max_model_len") is not None
                else None
            )
        ),
        gpu_memory_utilization=(
            args.target_gpu_memory_utilization
            if "--target_gpu_memory_utilization" in sys.argv
            else (
                float(default_target_mapping["gpu_memory_utilization"])
                if default_target_mapping.get("gpu_memory_utilization") is not None
                else None
            )
        ),
        request_interval_seconds=float(default_target_mapping.get("request_interval_seconds", 0.0)),
    )
    if not target_model.provider or not target_model.model:
        raise ValueError(f"Target model is not configured. Update {args.config_path} or pass --target_model/--target_provider.")

    judge_model = _build_model_config(
        default_mapping=default_judge_mapping or default_refuse_judge_mapping or default_attack_judge_mapping,
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
        max_tokens_override=judge_max_tokens_override,
        max_model_len=args.judge_max_model_len,
        max_model_len_flag="--judge_max_model_len",
        gpu_memory_utilization=args.judge_gpu_memory_utilization,
        gpu_memory_flag="--judge_gpu_memory_utilization",
    )
    if judge_model is None or not judge_model.provider or not judge_model.model:
        raise ValueError(f"Judge model is not configured. Update {args.config_path} or pass --judge_model/--judge_provider.")

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
        max_tokens_override=judge_max_tokens_override,
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
        max_tokens_override=judge_max_tokens_override,
        max_model_len=args.judge_max_model_len,
        max_model_len_flag="--judge_max_model_len",
        gpu_memory_utilization=args.judge_gpu_memory_utilization,
        gpu_memory_flag="--judge_gpu_memory_utilization",
    )

    attack_max_tokens = args.target_max_tokens if "--target_max_tokens" in sys.argv else attack_max_tokens_override
    refuse_max_tokens = args.target_max_tokens if "--target_max_tokens" in sys.argv else refuse_max_tokens_override

    print(
        f"[config] speed_profile={args.speed_profile}, attack_methods={list(attack_methods)}, "
        f"human_jailbreak_subsets={human_jailbreak_subsets}, attack_max_tokens={attack_max_tokens}, "
        f"refuse_max_tokens={refuse_max_tokens}, attack_judge_max_tokens={attack_judge_model.max_tokens if attack_judge_model else judge_model.max_tokens}, "
        f"refuse_judge_max_tokens={refuse_judge_model.max_tokens if refuse_judge_model else judge_model.max_tokens}"
    )

    run_redeval_redbench(
        RedEvalRunConfig(
            experiment_name=args.experiment_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            target_model=target_model,
            judge_model=judge_model,
            attack_judge_model=attack_judge_model,
            refuse_judge_model=refuse_judge_model,
            benchmarks=harmful_benchmarks | benign_benchmarks,
            harmful_benchmarks=harmful_benchmarks,
            benign_benchmarks=benign_benchmarks,
            benchmark_limits=_parse_benchmark_limits(args.benchmark_limits),
            dataset_truncate=args.dataset_truncate,
            experience_file=args.experience_file,
            experience_top_k=args.experience_top_k,
            experience_token_budget=args.experience_token_budget,
            attack_methods=attack_methods,
            human_jailbreak_subsets=human_jailbreak_subsets,
            attack_max_tokens=attack_max_tokens,
            refuse_max_tokens=refuse_max_tokens,
        )
    )


if __name__ == "__main__":
    main()
