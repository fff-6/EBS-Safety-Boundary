import argparse
import asyncio
import copy
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# Allow running via `python ebs/main.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ebs.llm import LLM
from ebs.core.experience_bank import (
    normalize_experience_bank,
)
from ebs.runtime.agents import SimpleAgent
from ebs.runtime.agents.common import TaskRecorder
from ebs.runtime.config import ConfigLoader
from ebs.runtime.utils import AgentsUtils


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


def _looks_like_qwen_model(model_name: str | None) -> bool:
    return bool(model_name) and str(model_name).strip().lower().startswith("qwen/")


def _apply_qwen_judge_defaults(args: argparse.Namespace) -> None:
    """Default Qwen judges to SiliconFlow when base URL / key are not passed explicitly."""

    if not _looks_like_qwen_model(args.judge_model):
        return
    if not args.judge_base_url:
        os.environ.setdefault("EBS_JUDGE_BASE_URL", os.getenv("TARGET_LLM_BASE_URL", "https://api.siliconflow.cn/v1"))
    if not args.judge_api_key:
        os.environ.setdefault("EBS_JUDGE_API_KEY", os.getenv("TARGET_LLM_API_KEY", ""))


def resolve_target_model_overrides() -> dict[str, str | float | None]:
    """Resolve optional target-model overrides from dedicated environment variables."""

    timeout = os.getenv("TARGET_LLM_TIMEOUT")
    return {
        "model": os.getenv("TARGET_LLM_MODEL"),
        "base_url": os.getenv("TARGET_LLM_BASE_URL"),
        "api_key": os.getenv("TARGET_LLM_API_KEY"),
        "timeout": float(timeout) if timeout else None,
    }


def load_rollouts(rollout_filename: str) -> list[dict]:
    results = []
    if os.path.exists(rollout_filename):
        with open(rollout_filename, encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
    return results


def save_rollouts(results: list[dict], rollout_filename: str):
    with open(rollout_filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def _build_prompt_trajectory(prompt: str, response: str, usage: dict | None = None) -> list[dict]:
    assistant_message = {"role": "assistant", "content": response}
    if usage is not None:
        assistant_message["usage"] = {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens")),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens")),
            "total_tokens": usage.get("total_tokens"),
        }
    return [
        {
            "trajectory": [
                {"role": "user", "content": prompt},
                assistant_message,
            ]
        }
    ]


async def rollout_dataset(
    worker_agent: SimpleAgent | None,
    data: list[dict],
    rollouts: list[dict],
    rollout_filename: str,
    verify_func: callable,
    skip_verify: bool = False,
    rollout_model: str | None = None,
    rollout_base_url: str | None = None,
    rollout_api_key: str | None = None,
    rollout_concurrency: int = 5,
    task_timeout: float = 3600,
    max_retries: int = 3,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> list[dict]:
    """Rollout the dataset using the worker agent with concurrency control, timeout, error handling, and retries."""

    # examine data and existing rollouts
    if len(rollouts) > 0:
        for each in rollouts:
            assert "runid" in each
        data_problems = [each["problem"] for each in data]
        rollouts_problems = [each["problem"] for each in rollouts]
        assert data_problems == rollouts_problems, (
            f"The problems in data should be the same as existing rollouts {rollout_filename}"
        )
    else:
        for sample in data:
            assert "problem" in sample and "groundtruth" in sample
        rollouts = [{"runid": i, **sample} for i, sample in enumerate(data)]
    save_rollouts(rollouts, rollout_filename)

    # create task queue
    task_queue = asyncio.Queue()
    pending_tasks_count = 0
    for sample in rollouts:
        if "trajectories" not in sample or len(sample["trajectories"]) == 0:
            sample_with_retry = copy.deepcopy(sample)
            sample_with_retry["retry_count"] = 0
            await task_queue.put(sample_with_retry)
            pending_tasks_count += 1
    pbar = tqdm(total=pending_tasks_count, desc="Rolling out")

    async def worker(name: str):
        while not task_queue.empty():
            sample = await task_queue.get()
            task_start_time = time.time()
            try:
                if worker_agent is None:
                    llm = LLM(model=rollout_model, base_url=rollout_base_url, api_key=rollout_api_key)
                    coro = asyncio.to_thread(
                        llm.chat,
                        sample["prompt"],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        return_usage=True,
                    )
                    response_text, usage = await asyncio.wait_for(coro, timeout=task_timeout)
                    res = TaskRecorder(
                        final_output=response_text,
                        trajectories=_build_prompt_trajectory(sample["prompt"], response_text, usage),
                    )
                else:
                    async with worker_agent as agent:
                        sample["trace_id"] = AgentsUtils.gen_trace_id()

                        async def rollout_streamed(sample) -> TaskRecorder:
                            prompt = sample.get("prompt", sample["problem"])
                            res = agent.run_streamed(prompt, trace_id=sample["trace_id"])
                            async for _ in res.stream_events():
                                pass
                            traj = AgentsUtils.get_trajectory_from_agent_result(res)
                            return TaskRecorder(
                                final_output=res.final_output,
                                trajectories=[traj],
                            )

                        res = await asyncio.wait_for(rollout_streamed(sample), timeout=task_timeout)

                task_end_time = time.time()
                sample.update(
                    {
                        "response": res.final_output,
                        "trajectories": res.trajectories,
                        "error": None,
                        "rollout_time": task_end_time - task_start_time,
                        "online_metrics": sample.get("online_metrics", {}),
                    }
                )
                sample["reward"] = 0.0 if skip_verify else verify_func(sample, sample["groundtruth"])

                # Task succeeded
                rollouts[sample["runid"]] = sample
                save_rollouts(rollouts, rollout_filename)
                pbar.update(1)

            except Exception as e:
                task_end_time = time.time()
                sample["retry_count"] += 1
                error_info = traceback.format_exc()
                print(f"> error: {error_info}")

                if sample["retry_count"] <= max_retries:
                    tqdm.write(
                        f"Worker {name}: Task runid={sample['runid']} failed with {type(e).__name__}. Retrying ({sample['retry_count']}/{max_retries})..."
                    )
                    await task_queue.put(sample)  # Re-queue the task
                else:
                    tqdm.write(
                        f"Worker {name}: Task runid={sample['runid']} failed after {max_retries} retries. Error: {e}. Traceback: {error_info}"
                    )
                    sample.update(
                        {
                            "response": f"Error: {str(e)} after {max_retries} retries.",
                            "trajectories": [],
                            "error": error_info,
                            "reward": 0,
                            "rollout_time": task_end_time - task_start_time,
                            "online_metrics": sample.get("online_metrics", {}),
                        }
                    )

                    # Task failed permanently
                    rollouts[sample["runid"]] = sample
                    save_rollouts(rollouts, rollout_filename)
                    pbar.update(1)
            finally:
                task_queue.task_done()

    # run all tasks
    workers = [asyncio.create_task(worker(f"worker-{i}")) for i in range(rollout_concurrency)]
    await task_queue.join()

    # clean up
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    pbar.close()
    print(f"Successfully processed {len(rollouts)} samples.")

    # stats
    all_rewards = []
    problem_to_scores = defaultdict(list)
    num_tool_calls = []
    for rollout in rollouts:
        all_rewards.append(rollout.get("reward", 0))
        problem_to_scores[rollout["problem"]].append(rollout.get("reward", 0))
        if "trajectories" in rollout and rollout["trajectories"]:
            num_tool_calls.append(
                len([each for each in rollout["trajectories"][0]["trajectory"] if each["role"] == "tool"])
            )
    problem_to_max_score = {problem: max(scores) for problem, scores in problem_to_scores.items()}
    max_K = max((len(scores) for scores in problem_to_scores.values()), default=0)
    stats = {
        "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
        f"Pass@{max_K}": sum(max_reward > 0 for max_reward in problem_to_max_score.values()) / len(problem_to_max_score)
        if problem_to_max_score
        else 0,
        "avg_tool_call": sum(num_tool_calls) / len(num_tool_calls) if num_tool_calls else 0,
    }
    for k, v in stats.items():
        print(f"- {k}: {v}")
    return rollouts, stats


async def main(args):
    _load_project_env()
    _apply_qwen_judge_defaults(args)

    if args.judge_model:
        os.environ["EBS_JUDGE_MODEL"] = args.judge_model
    if args.judge_base_url:
        os.environ["EBS_JUDGE_BASE_URL"] = args.judge_base_url
    if args.judge_api_key:
        os.environ["EBS_JUDGE_API_KEY"] = args.judge_api_key

    target_overrides = resolve_target_model_overrides()

    if args.domain != "ebs":
        raise ValueError(f"Unsupported domain: {args.domain}. Only `ebs` is supported in this repository.")

    from ebs.core.dataset import load_data
    from ebs.core.prompts import build_ebs_prompt_with_metrics
    from ebs.core.verify import verify_func

    config_name = "simple/ebs_agent.yaml"

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        rollout_model = args.rollout_model or target_overrides["model"]
        rollout_base_url = args.rollout_base_url or target_overrides["base_url"]
        rollout_api_key = args.rollout_api_key or target_overrides["api_key"]
        client_timeout = args.client_timeout if args.client_timeout is not None else target_overrides["timeout"]
        if rollout_model:
            config.model.model_provider.model = str(rollout_model)
        if rollout_base_url:
            config.model.model_provider.base_url = str(rollout_base_url)
        if rollout_api_key:
            config.model.model_provider.api_key = str(rollout_api_key)
        if client_timeout is not None:
            config.model.model_provider.timeout = float(client_timeout)
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset
    test_data = load_data(args.dataset)
    print(f"Loaded {len(test_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- truncated to {args.dataset_truncate}")
        test_data = test_data[: args.dataset_truncate]

    # Insert experiences
    if args.experience_file:
        experiences = normalize_experience_bank(json.load(open(args.experience_file, encoding="utf-8")))
        formatted_test_data = []
        for each in test_data:
            prompt, prompt_metrics = build_ebs_prompt_with_metrics(
                each["problem"],
                experiences=experiences,
                disable_experience_retrieval=args.disable_experience_retrieval,
            )
            formatted_test_data.append(
                {
                    "prompt": prompt,
                    "online_metrics": prompt_metrics,
                    **each,
                }
            )
    else:
        formatted_test_data = [
            {
                "prompt": each["problem"],
                "online_metrics": {
                    "selected_bucket": None,
                    "routing_ms": 0.0,
                    "retrieval_ms": 0.0,
                    "route_confidence": None,
                    "num_selected_experiences": 0,
                },
                **each,
            }
            for each in test_data
        ]

    # Duplicate for Pass@k evaluation
    formatted_test_data = formatted_test_data * args.pass_k
    print(f"Duplicated to {len(formatted_test_data)} records for Pass@{args.pass_k} evaluation")

    # Load existing rollouts
    os.makedirs(f"data/{args.domain}/eval", exist_ok=True)
    rollout_filename = f"data/{args.domain}/eval/{args.experiment_name}.jsonl"
    rollouts = load_rollouts(rollout_filename)

    # Rollout the dataset
    await rollout_dataset(
        worker_agent=worker_agent,
        data=formatted_test_data,
        rollouts=rollouts,
        verify_func=verify_func,
        rollout_filename=rollout_filename,
        skip_verify=args.skip_verify,
        rollout_model=args.rollout_model or target_overrides["model"],
        rollout_base_url=args.rollout_base_url or target_overrides["base_url"],
        rollout_api_key=args.rollout_api_key or target_overrides["api_key"],
        rollout_concurrency=args.rollout_concurrency,
        task_timeout=args.task_timeout,
        max_tokens=args.rollout_max_tokens,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EBS experience reuse and evaluation")
    parser.add_argument(
        "--mode", type=str, default="agent", required=True, choices=["prompt", "agent"], help="Mode of inference"
    )
    parser.add_argument(
        "--domain", type=str, required=True, choices=["ebs"], help="The domain of the experiment"
    )
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment run")
    parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
    parser.add_argument("--dataset_truncate", type=int, default=None, help="Truncate dataset to first N samples")
    parser.add_argument("--experience_file", type=str, default=None)
    parser.add_argument("--skip_verify", action="store_true", help="Skip judge-model verification to measure raw inference cost only.")
    parser.add_argument(
        "--disable_experience_retrieval",
        action="store_true",
        help="Build the EBS base prompt with routing but without injecting retrieved experiences.",
    )
    parser.add_argument("--rollout_concurrency", type=int, default=5, help="Concurrency level for rollouts")
    parser.add_argument("--rollout_max_tokens", type=int, default=4096, help="Max tokens for each rollout")
    parser.add_argument("--pass_k", type=int, default=1, help="Pass@k metric")
    parser.add_argument("--task_timeout", type=float, default=3600, help="Timeout for each individual task in seconds")
    parser.add_argument(
        "--client_timeout",
        type=float,
        default=None,
        help="Timeout in seconds for each underlying OpenAI-compatible HTTP request.",
    )
    parser.add_argument("--rollout_model", type=str, default=None, help="Override rollout model for evaluation.")
    parser.add_argument("--rollout_base_url", type=str, default=None, help="Override rollout model base URL.")
    parser.add_argument("--rollout_api_key", type=str, default=None, help="Override rollout model API key.")
    parser.add_argument("--judge_model", type=str, default=None, help="Override judge model.")
    parser.add_argument("--judge_base_url", type=str, default=None, help="Override judge model base URL.")
    parser.add_argument("--judge_api_key", type=str, default=None, help="Override judge model API key.")
    args = parser.parse_args()
    asyncio.run(main(args))
