from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe_parse_tool_content(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def summarize_rollouts(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    total_samples = 0
    total_extra_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_s = 0.0
    routing_values: list[float] = []
    retrieval_values: list[float] = []

    for row in rows:
        trajectories = row.get("trajectories") or []
        if not trajectories:
            continue
        trajectory = trajectories[0].get("trajectory") or []
        usage_messages = [
            item.get("usage")
            for item in trajectory
            if item.get("role") == "assistant" and isinstance(item.get("usage"), dict)
        ]
        total_samples += 1
        total_extra_calls += max(0, len(usage_messages) - 1)
        total_input_tokens += sum(int(msg.get("input_tokens") or 0) for msg in usage_messages)
        total_output_tokens += sum(int(msg.get("output_tokens") or 0) for msg in usage_messages)
        total_latency_s += float(row.get("rollout_time") or 0.0)
        online_metrics = row.get("online_metrics") or {}
        if online_metrics.get("routing_ms") is not None:
            routing_values.append(float(online_metrics["routing_ms"]))
        if online_metrics.get("retrieval_ms") is not None:
            retrieval_values.append(float(online_metrics["retrieval_ms"]))

        for item in trajectory:
            if item.get("role") != "tool":
                continue
            parsed = _safe_parse_tool_content(item.get("content"))
            if not parsed:
                continue
            timing = parsed.get("timing_ms")
            if isinstance(timing, dict):
                if timing.get("assessment") is not None:
                    routing_values.append(float(timing["assessment"]))
                if timing.get("retrieval") is not None:
                    retrieval_values.append(float(timing["retrieval"]))

    if total_samples == 0:
        raise ValueError("No valid rollout samples with trajectories were found.")

    return {
        "samples": total_samples,
        "extra_calls": total_extra_calls / total_samples,
        "avg_input_tokens": total_input_tokens / total_samples,
        "avg_output_tokens": total_output_tokens / total_samples,
        "avg_routing_ms": (sum(routing_values) / len(routing_values)) if routing_values else 0.0,
        "avg_retrieval_ms": (sum(retrieval_values) / len(retrieval_values)) if retrieval_values else 0.0,
        "avg_latency_s": total_latency_s / total_samples,
    }


def compute_cost(avg_input_tokens: float, avg_output_tokens: float, input_price: float, output_price: float) -> float:
    return (avg_input_tokens / 1_000_000.0) * input_price + (avg_output_tokens / 1_000_000.0) * output_price


def format_metric(value: float | None, digits: int = 2, missing: str = "—") -> str:
    if value is None:
        return missing
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Vanilla vs EBS online inference cost from rollout jsonl files.")
    parser.add_argument("--vanilla_rollout", type=Path, required=True)
    parser.add_argument("--ebs_no_retrieval_rollout", type=Path, default=None)
    parser.add_argument("--ebs_rollout", type=Path, required=True)
    parser.add_argument("--input_price_per_1m", type=float, default=1.0)
    parser.add_argument("--output_price_per_1m", type=float, default=1.0)
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args()

    vanilla = summarize_rollouts(load_jsonl(args.vanilla_rollout))
    ebs_no_retrieval = (
        summarize_rollouts(load_jsonl(args.ebs_no_retrieval_rollout)) if args.ebs_no_retrieval_rollout is not None else None
    )
    ebs = summarize_rollouts(load_jsonl(args.ebs_rollout))

    vanilla_cost = compute_cost(
        float(vanilla["avg_input_tokens"]),
        float(vanilla["avg_output_tokens"]),
        args.input_price_per_1m,
        args.output_price_per_1m,
    )
    ebs_cost = compute_cost(
        float(ebs["avg_input_tokens"]),
        float(ebs["avg_output_tokens"]),
        args.input_price_per_1m,
        args.output_price_per_1m,
    )

    summary: dict[str, dict[str, float | None]] = {
        "Vanilla": {
            **vanilla,
            "cost_ratio": 1.0,
            "estimated_cost_per_sample": vanilla_cost,
        }
    }
    if ebs_no_retrieval is not None:
        ebs_no_retrieval_cost = compute_cost(
            float(ebs_no_retrieval["avg_input_tokens"]),
            float(ebs_no_retrieval["avg_output_tokens"]),
            args.input_price_per_1m,
            args.output_price_per_1m,
        )
        summary["EBS w/o Retrieval"] = {
            **ebs_no_retrieval,
            "cost_ratio": (ebs_no_retrieval_cost / vanilla_cost) if vanilla_cost > 0 else None,
            "estimated_cost_per_sample": ebs_no_retrieval_cost,
        }
    summary["EBS"] = {
        **ebs,
        "cost_ratio": (ebs_cost / vanilla_cost) if vanilla_cost > 0 else None,
        "estimated_cost_per_sample": ebs_cost,
    }

    def print_row(name: str, metrics: dict[str, float | None], ratio: float | None) -> None:
        print(
            f"| {name} | "
            f"{format_metric(float(metrics['extra_calls']))} | "
            f"{format_metric(float(metrics['avg_input_tokens']))} | "
            f"{format_metric(float(metrics['avg_output_tokens']))} | "
            f"{format_metric(float(metrics['avg_routing_ms']))} | "
            f"{format_metric(float(metrics['avg_retrieval_ms']))} | "
            f"{format_metric(float(metrics['avg_latency_s']), digits=3)} | "
            f"{format_metric(float(ratio) if ratio is not None else None)}x |"
        )

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("| Method | Extra LLM Calls | Avg. Input Tok. | Avg. Output Tok. | Routing / ms | Retrieval / ms | Latency / s | Cost Ratio |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    print_row("Vanilla", summary["Vanilla"], summary["Vanilla"]["cost_ratio"])
    if ebs_no_retrieval is not None:
        print_row("EBS w/o Retrieval", summary["EBS w/o Retrieval"], summary["EBS w/o Retrieval"]["cost_ratio"])
    print_row("EBS", summary["EBS"], summary["EBS"]["cost_ratio"])


if __name__ == "__main__":
    main()
