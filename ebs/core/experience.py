import copy
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ebs.llm import LLM
from ebs.core.experience_bank import (
    CATEGORY_KEYS,
    format_experiences_for_prompt,
    get_next_experience_id,
    normalize_experience_bank,
)
from ebs.core.prompts import (
    BATCH_EXPERIENCE_UPDATE_TEMPLATE,
    ETHICS_BATCH_EXPERIENCE_UPDATE_TEMPLATE,
    ETHICS_SINGLE_QUERY_CRITIQUE_TEMPLATE,
    ETHICS_SINGLE_ROLLOUT_EXPERIENCE_TEMPLATE,
    ETHICS_SINGLE_ROLLOUT_SUMMARY_TEMPLATE,
    SINGLE_QUERY_CRITIQUE_TEMPLATE,
    SINGLE_ROLLOUT_EXPERIENCE_TEMPLATE,
    SINGLE_ROLLOUT_SUMMARY_TEMPLATE,
)


class ExperienceUpdater:
    def __init__(self):
        self.llm = LLM()
        self._extraction_max_retries = 3

    _compiled_cjk = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

    def _contains_cjk(self, text: str) -> bool:
        return bool(self._compiled_cjk.search(text or ""))

    def _rewrite_experience_in_english(self, text: str, bucket: str) -> str:
        if not self._contains_cjk(text):
            return text
        domain = "ethics" if bucket == "ethics" else "safety"
        prompt = (
            f"Rewrite the following {domain} experience as concise English only.\n"
            "- Keep the original meaning.\n"
            "- Return one sentence only.\n"
            "- Do not use Chinese or any other non-English language.\n"
            "- Keep it as a reusable policy or reasoning guideline.\n"
            "- Prefer under 32 words.\n\n"
            f"Experience:\n{text}"
        )
        rewritten = self.llm.chat(prompt).strip()
        return rewritten or text

    def _normalize_operation_language(self, operation: dict, bucket: str) -> dict:
        normalized = copy.deepcopy(operation)
        experience = normalized.get("experience")
        if isinstance(experience, str):
            normalized["experience"] = self._rewrite_experience_in_english(experience, bucket)
        return normalized

    def _normalize_bucket_language(self, experiences: dict[str, str], bucket: str) -> dict[str, str]:
        normalized_bucket: dict[str, str] = {}
        for exp_id, text in experiences.items():
            normalized_bucket[exp_id] = self._rewrite_experience_in_english(text, bucket) if isinstance(text, str) else text
        return normalized_bucket

    def _parse_json_payload(self, response: str):
        payload = response.split("```json")[-1].split("```")[0].strip()
        return json.loads(payload)

    def _chat_json_with_retries(self, prompt: str, *, context: str):
        last_error: Exception | None = None
        for attempt in range(1, self._extraction_max_retries + 1):
            try:
                response = self.llm.chat(prompt)
                return response, self._parse_json_payload(response)
            except Exception as exc:
                last_error = exc
                if attempt < self._extraction_max_retries:
                    print(
                        f"Warning: {context} parse failed on attempt "
                        f"{attempt}/{self._extraction_max_retries}, retrying: {exc}"
                    )
        assert last_error is not None
        raise last_error

    def run(
        self,
        rollouts,
        experiences,
        save_dir,
        max_workers=16,
        given_ground_truth=True,
        only_partial_correct=True,
        experience_update_method="critique",
    ):
        del given_ground_truth
        experiences = normalize_experience_bank(experiences)
        problem_to_summarized_rollouts = self._single_rollout_summary(
            rollouts=rollouts,
            save_dir=save_dir,
            max_workers=max_workers,
            only_partial_correct=only_partial_correct,
        )
        if experience_update_method == "critique":
            critiques = self._single_query_critique(
                problem_to_summarized_rollouts=problem_to_summarized_rollouts,
                experiences=experiences,
                save_dir=save_dir,
                max_workers=max_workers,
                only_partial_correct=only_partial_correct,
            )
        elif experience_update_method == "summary":
            critiques = self._single_rollout_experience(
                problem_to_summarized_rollouts=problem_to_summarized_rollouts,
                experiences=experiences,
                save_dir=save_dir,
                max_workers=max_workers,
            )
        else:
            raise ValueError(
                f"Unsupported experience_update_method: {experience_update_method}. "
                "Expected one of: critique, summary."
            )
        new_experiences = self._batch_update(
            experiences=experiences,
            critiques=critiques,
            save_dir=save_dir,
            experience_update_method=experience_update_method,
        )
        return new_experiences

    def _single_rollout_summary(self, rollouts, save_dir, max_workers, only_partial_correct=True):
        filename = os.path.join(save_dir, "single_rollout_summary.json")
        if os.path.exists(filename):
            with open(filename, encoding="utf-8") as f:
                results = json.load(f)
                if results:
                    print("Single rollout summary")
                    print("- File exists, loaded from:", filename)
                    return results

        problems_to_rollouts = defaultdict(list)
        for each in rollouts:
            if "trajectories" in each and each["trajectories"]:
                problems_to_rollouts[each["problem"]].append(each)
        results = defaultdict(list)

        all_rollouts_to_process = []
        for grouped_rollouts in problems_to_rollouts.values():
            if only_partial_correct:
                scores = [each.get("reward", 0) for each in grouped_rollouts]
                avg_score = sum(scores) / len(scores)
                if 0 < avg_score < 1:
                    all_rollouts_to_process.extend(grouped_rollouts)
            else:
                all_rollouts_to_process.extend(grouped_rollouts)

        def process(cur):
            try:
                judge = cur.get("judge", {})
                bucket = cur.get("groundtruth", {}).get("experience_bucket", "harmful")
                template = ETHICS_SINGLE_ROLLOUT_SUMMARY_TEMPLATE if bucket == "ethics" else SINGLE_ROLLOUT_SUMMARY_TEMPLATE
                response = self.llm.chat(
                    template.format(
                        problem=cur["problem"],
                        trajectory=cur["trajectories"][0]["trajectory"],
                        expected_behavior=judge.get("expected_behavior", "Calibrate refusal correctly."),
                        reward=cur.get("reward", 0),
                        general_reward=judge.get("general_reward", 0),
                        risk_reward=judge.get("risk_reward", 0),
                    )
                )
                return {"trajectory_summary": response, **cur}
            except Exception as e:
                print(f"Warning: failed in single rollout summary, {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_rollout = {executor.submit(process, cur): cur for cur in all_rollouts_to_process}
            for future in tqdm(
                as_completed(future_to_rollout), total=len(all_rollouts_to_process), desc="Single rollout summary"
            ):
                result = future.result()
                if result is not None:
                    results[result["problem"]].append(result)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    def _single_query_critique(
        self,
        problem_to_summarized_rollouts,
        experiences,
        save_dir,
        max_workers,
        max_operations=1,
        only_partial_correct=True,
    ):
        filename = os.path.join(save_dir, "single_query_critique.json")
        if os.path.exists(filename):
            with open(filename, encoding="utf-8") as f:
                results = json.load(f)
                if results:
                    print("Single query critique")
                    print("- File exists, loaded from:", filename)
                    return results

        all_rollouts = []
        for grouped_rollouts in problem_to_summarized_rollouts.values():
            if only_partial_correct:
                scores = [each.get("reward", 0) for each in grouped_rollouts]
                avg_score = sum(scores) / len(scores)
                if 0 < avg_score < 1:
                    all_rollouts.append(grouped_rollouts)
            else:
                all_rollouts.append(grouped_rollouts)

        def process(rollouts_per_problem):
            try:
                problem = rollouts_per_problem[0]["problem"]
                bucket = rollouts_per_problem[0].get("groundtruth", {}).get("experience_bucket", "harmful")
                request_type = bucket if bucket == "ethics" else ("harmful" if rollouts_per_problem[0]["judge"]["harmful_label"] else "benign")
                formatted_attempts = "\n\n".join(
                    [
                        "Attempt {idx} (reward={reward:.2f}, general={general:.2f}, risk={risk:.2f}):\n{summary}".format(
                            idx=i + 1,
                            reward=each.get("reward", 0),
                            general=each.get("judge", {}).get("general_reward", 0),
                            risk=each.get("judge", {}).get("risk_reward", 0),
                            summary=each["trajectory_summary"],
                        )
                        for i, each in enumerate(rollouts_per_problem)
                    ]
                )
                formatted_experiences = format_experiences_for_prompt(experiences.get(bucket, {}))
                template = ETHICS_SINGLE_QUERY_CRITIQUE_TEMPLATE if bucket == "ethics" else SINGLE_QUERY_CRITIQUE_TEMPLATE
                response = self.llm.chat(
                    template.format(
                        max_operations=max_operations,
                        problem=problem,
                        request_type=request_type,
                        trajectories=formatted_attempts,
                        experiences=formatted_experiences,
                    )
                )
                operations = self._parse_json_payload(response)
                operations = [self._normalize_operation_language(operation, bucket) for operation in operations]
                return {
                    "bucket": bucket,
                    "rollouts": rollouts_per_problem,
                    "critique": response,
                    "operations": operations[:max_operations],
                }
            except Exception as e:
                print(f"Warning: failed in single query critique, {e}")
                return None

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {
                executor.submit(process, rollouts_per_problem): rollouts_per_problem
                for rollouts_per_problem in all_rollouts
            }
            for future in tqdm(as_completed(future_to_case), total=len(all_rollouts), desc="Single query critique"):
                result = future.result()
                if result is not None:
                    results.append(result)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    def _single_rollout_experience(
        self,
        problem_to_summarized_rollouts,
        experiences,
        save_dir,
        max_workers,
        max_operations=1,
    ):
        filename = os.path.join(save_dir, "single_rollout_experience.json")
        if os.path.exists(filename):
            with open(filename, encoding="utf-8") as f:
                results = json.load(f)
                if results:
                    print("Single rollout experience")
                    print("- File exists, loaded from:", filename)
                    return results

        all_rollouts = [
            rollout
            for grouped_rollouts in problem_to_summarized_rollouts.values()
            for rollout in grouped_rollouts
        ]

        def process(rollout):
            try:
                problem = rollout["problem"]
                judge = rollout.get("judge", {})
                bucket = rollout.get("groundtruth", {}).get("experience_bucket", "harmful")
                request_type = bucket if bucket == "ethics" else ("harmful" if judge["harmful_label"] else "benign")
                formatted_experiences = format_experiences_for_prompt(experiences.get(bucket, {}))
                template = (
                    ETHICS_SINGLE_ROLLOUT_EXPERIENCE_TEMPLATE
                    if bucket == "ethics"
                    else SINGLE_ROLLOUT_EXPERIENCE_TEMPLATE
                )
                response, operations = self._chat_json_with_retries(
                    template.format(
                        max_operations=max_operations,
                        problem=problem,
                        request_type=request_type,
                        expected_behavior=judge.get("expected_behavior", "Calibrate refusal correctly."),
                        reward=rollout.get("reward", 0),
                        general_reward=judge.get("general_reward", 0),
                        risk_reward=judge.get("risk_reward", 0),
                        trajectory_summary=rollout["trajectory_summary"],
                        experiences=formatted_experiences,
                    ),
                    context="single rollout experience extraction",
                )
                operations = [self._normalize_operation_language(operation, bucket) for operation in operations]
                return {
                    "bucket": bucket,
                    "rollout": rollout,
                    "experience_extraction": response,
                    "operations": operations[:max_operations],
                }
            except Exception as e:
                print(f"Warning: failed in single rollout experience extraction, {e}")
                return None

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_rollout = {executor.submit(process, rollout): rollout for rollout in all_rollouts}
            for future in tqdm(
                as_completed(future_to_rollout), total=len(all_rollouts), desc="Single rollout experience"
            ):
                result = future.result()
                if result is not None:
                    results.append(result)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    def _batch_update(self, experiences, critiques, save_dir, max_retries=3, experience_update_method="critique"):
        print("Batch update")
        filename = (
            os.path.join(save_dir, "batch_update.json")
            if experience_update_method == "critique"
            else os.path.join(save_dir, f"batch_update_{experience_update_method}.json")
        )
        if os.path.exists(filename):
            results = json.load(open(filename, encoding="utf-8"))
            print("- File exists, loaded from:", filename)
            if "new_experiences" in results:
                return normalize_experience_bank(results["new_experiences"])
            return normalize_experience_bank(results)

        critiques_by_bucket = {category: [] for category in CATEGORY_KEYS}
        for each in critiques:
            bucket = each.get("bucket", "harmful")
            if bucket in critiques_by_bucket:
                critiques_by_bucket[bucket].append(each)

        all_operations: dict[str, list[dict]] = {category: [] for category in CATEGORY_KEYS}
        responses: dict[str, str] = {}
        revision_plans: dict[str, list[dict]] = {category: [] for category in CATEGORY_KEYS}
        new_experiences = copy.deepcopy(experiences)

        for bucket in CATEGORY_KEYS:
            for each in critiques_by_bucket[bucket]:
                try:
                    all_operations[bucket].extend(each["operations"])
                except Exception:
                    print(f"Warning: failed to decode operation: {each}")

            candidate_experiences = copy.deepcopy(new_experiences[bucket])
            to_modify = []
            for operation in all_operations[bucket]:
                try:
                    if operation["option"] == "modify":
                        if operation["modified_from"] in candidate_experiences:
                            to_modify.append(operation)
                    elif operation["option"] == "add":
                        next_id = get_next_experience_id(candidate_experiences, bucket)
                        candidate_experiences[next_id] = operation["experience"]
                except Exception:
                    print(f"Warning: failed to decode operation: {operation}")

            template = ETHICS_BATCH_EXPERIENCE_UPDATE_TEMPLATE if bucket == "ethics" else BATCH_EXPERIENCE_UPDATE_TEMPLATE
            response = "[]"
            revision_plan = []
            if candidate_experiences or to_modify:
                for _ in range(max_retries):
                    try:
                        response, revision_plan = self._chat_json_with_retries(
                            template.format(experiences=candidate_experiences, updates=to_modify),
                            context=f"batch update for bucket={bucket}",
                        )
                        break
                    except Exception:
                        print(f"Warning: failed to decode in updating EBS experiences for bucket={bucket}")

            revised_bucket = copy.deepcopy(candidate_experiences)
            for operation in revision_plan:
                try:
                    if operation["option"] == "modify":
                        revised_bucket[operation["modified_from"]] = operation["experience"]
                    elif operation["option"] == "merge":
                        valid_ids = [mid for mid in operation["merged_from"] if mid in revised_bucket]
                        for exp_id in valid_ids:
                            del revised_bucket[exp_id]
                        next_id = get_next_experience_id(revised_bucket, bucket)
                        revised_bucket[next_id] = operation["experience"]
                except Exception as e:
                    print("Error: failed to complete EBS experience update:", operation, "|", e)

            new_experiences[bucket] = self._normalize_bucket_language(revised_bucket, bucket)
            responses[bucket] = response
            revision_plans[bucket] = revision_plan

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experience_update_method": experience_update_method,
                    "operations": all_operations,
                    "response": responses,
                    "revision_plan": revision_plans,
                    "new_experiences": new_experiences,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return new_experiences
