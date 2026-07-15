import argparse
import asyncio
import copy
import json
import os
import random
import sys
from pathlib import Path

# Allow running via `python ebs/train.py`.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ebs.main import load_rollouts, rollout_dataset
from ebs.main import _load_project_env, resolve_target_model_overrides
from ebs.core.experience_bank import (
    has_any_experiences,
    normalize_experience_bank,
)
from ebs.runtime.agents import SimpleAgent
from ebs.runtime.config import ConfigLoader

random.seed(42)


async def main(args):
    _load_project_env()

    target_overrides = resolve_target_model_overrides()

    if args.domain != "ebs":
        raise ValueError(f"Unsupported domain: {args.domain}. Only `ebs` is supported in this repository.")

    from ebs.core.dataset import load_data
    from ebs.core.experience import ExperienceUpdater
    from ebs.core.prompts import build_ebs_prompt
    from ebs.core.verify import verify_func

    config_name = "simple/ebs_agent.yaml"

    # Create experiment directory
    experiment_dir = os.path.join("data", args.domain, "train", args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Set up the agent
    if args.mode == "prompt":
        worker_agent = None
    elif args.mode == "agent":
        config = ConfigLoader.load_agent_config(config_name)
        config.model.model_settings.temperature = args.rollout_temperature
        if target_overrides["model"]:
            config.model.model_provider.model = str(target_overrides["model"])
        if target_overrides["base_url"]:
            config.model.model_provider.base_url = str(target_overrides["base_url"])
        if target_overrides["api_key"]:
            config.model.model_provider.api_key = str(target_overrides["api_key"])
        client_timeout = args.client_timeout if args.client_timeout is not None else target_overrides["timeout"]
        if client_timeout is not None:
            config.model.model_provider.timeout = float(client_timeout)
        worker_agent = SimpleAgent(config=config)
        await worker_agent.build()
    else:
        raise ValueError(f"Unsupported inference mode: {args.mode}")

    # Load the dataset
    train_data = load_data(args.dataset)
    print(f"Loaded {len(train_data)} records from dataset")
    if args.dataset_truncate is not None:
        print(f"- truncated to {args.dataset_truncate}")
        train_data = train_data[: args.dataset_truncate]
    assert len(train_data) % args.batchsize == 0

    # Set up the stats
    stats_filename = os.path.join(experiment_dir, "stats.json")
    if os.path.exists(stats_filename):
        stats = json.load(open(stats_filename))
    else:
        stats = {}

    # Train
    for epoch in range(args.epochs):
        # Init
        print("=" * 30 + f"\nEpoch {epoch}\n" + "=" * 30)
        cur_epoch_dir = os.path.join(experiment_dir, f"epoch_{epoch}")
        os.makedirs(cur_epoch_dir, exist_ok=True)

        # Check if shuffled data already exists for this epoch
        shuffled_filename = os.path.join(cur_epoch_dir, "shuffled_data.jsonl")
        if os.path.exists(shuffled_filename):
            shuffled_data = []
            with open(shuffled_filename) as f:
                for line in f:
                    shuffled_data.append(json.loads(line))
            print(f"Loaded {len(shuffled_data)} records from shuffled data")
        else:
            print("Shuffling data ...")
            shuffled_data = copy.deepcopy(train_data)
            random.shuffle(shuffled_data)
            with open(shuffled_filename, "w") as f:
                for each in shuffled_data:
                    f.write(json.dumps(each) + "\n")

        # for each batch
        num_batches = len(shuffled_data) // args.batchsize
        for batch_idx in range(num_batches):
            step = epoch * num_batches + batch_idx
            if f"step_{step}" not in stats:
                stats[f"step_{step}"] = {"epoch": epoch, "batch": batch_idx, "complete": False}
            elif stats[f"step_{step}"]["complete"]:
                continue

            # Init
            print(f"Step {step} (Epoch {epoch}, Batch {batch_idx})")
            cur_step_dir = os.path.join(experiment_dir, f"step_{step}")
            os.makedirs(cur_step_dir, exist_ok=True)

            # Get current batch data
            batch_data = copy.deepcopy(shuffled_data[batch_idx * args.batchsize : (batch_idx + 1) * args.batchsize])

            # Load existing rollouts
            rollout_filename = os.path.join(cur_step_dir, "rollout.jsonl")
            rollouts = load_rollouts(rollout_filename)

            # Retrieve experiences for this batch (except first step)
            if step > 0:
                experience_filename = os.path.join(
                    "data", args.domain, "train", args.experiment_name, f"step_{step}/experiences.json"
                )
                experiences = normalize_experience_bank(json.load(open(experience_filename, encoding="utf-8")))
            else:
                experiences = normalize_experience_bank({})

            # Format the batch data with experiences
            formatted_batch_data = [
                {
                    "prompt": build_ebs_prompt(each["problem"], experiences=experiences)
                    if has_any_experiences(experiences)
                    else each["problem"],
                    **each,
                }
                for each in batch_data
            ]

            # Duplicate for GRPO
            print(f"GRPO rollout number={args.grpo_n}")
            formatted_batch_data = formatted_batch_data * args.grpo_n

            # Rollout the dataset
            rollouts, rollout_stats = await rollout_dataset(
                worker_agent=worker_agent,
                data=formatted_batch_data,
                rollouts=rollouts,
                verify_func=verify_func,
                rollout_filename=rollout_filename,
                rollout_concurrency=args.rollout_concurrency,
                task_timeout=args.task_timeout,
                temperature=args.rollout_temperature,
                max_tokens=args.rollout_max_tokens,
            )
            stats[f"step_{step}"]["rollout"] = rollout_stats

            # Generate critiques and update experiences
            next_step_dir = os.path.join(experiment_dir, f"step_{step + 1}")
            os.makedirs(next_step_dir, exist_ok=True)
            next_experience_filename = os.path.join(next_step_dir, "experiences.json")
            if os.path.exists(next_experience_filename):
                print(f"Experiences already exist for step {step}, skipping experience update")
            else:
                new_experiences = ExperienceUpdater().run(
                    rollouts=rollouts,
                    experiences=experiences,
                    save_dir=cur_step_dir,
                    max_workers=args.rollout_concurrency,
                    given_ground_truth=True if args.given_ground_truth == "True" else False,
                    only_partial_correct=True if args.grpo_n > 1 else False,
                    experience_update_method=args.experience_update_method,
                )
                json.dump(new_experiences, open(next_experience_filename, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                print(f"Saved {len(new_experiences)} experiences to {next_experience_filename}")

            # Save stats
            stats[f"step_{step}"]["complete"] = True
            json.dump(stats, open(stats_filename, "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EBS experience generation")
    parser.add_argument(
        "--mode", type=str, default="agent", required=True, choices=["prompt", "agent"], help="Mode of inference"
    )
    parser.add_argument(
        "--domain", type=str, required=True, choices=["ebs"], help="domain of the tasks (ebs)"
    )
    parser.add_argument("--experiment_name", type=str, required=True, help="name of experiment run")
    parser.add_argument("--dataset", type=str, required="True", help="Name of dataset")
    parser.add_argument("--dataset_truncate", type=int, default=None, help="Truncate dataset to first N samples")
    parser.add_argument("--given_ground_truth", type=str, default="True", help="Whether use ground truth answers")
    parser.add_argument("--epochs", type=int, default=2, help="number of training epochs")
    parser.add_argument("--batchsize", type=int, default=64, help="batchsize")
    parser.add_argument("--grpo_n", type=int, default=5, help="number of rollouts in a group of GRPO")
    parser.add_argument("--rollout_concurrency", type=int, default=5, help="Concurrency level for rollouts")
    parser.add_argument("--rollout_temperature", type=float, default=0.7, help="Temperature for the LLM")
    parser.add_argument("--rollout_max_tokens", type=int, default=4096, help="Max tokens for each rollout batch")
    parser.add_argument("--task_timeout", type=float, default=3600, help="Timeout for each individual task in seconds")
    parser.add_argument(
        "--experience_update_method",
        type=str,
        default="critique",
        choices=["critique", "summary"],
        help="Experience update strategy: compare multiple attempts (`critique`) or extract from single-rollout summaries (`summary`).",
    )
    parser.add_argument(
        "--client_timeout",
        type=float,
        default=None,
        help="Timeout in seconds for each underlying OpenAI-compatible HTTP request.",
    )
    args = parser.parse_args()
    asyncio.run(main(args))
