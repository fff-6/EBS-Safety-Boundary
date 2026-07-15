import time

import openai
import httpx

from ebs.runtime.utils import EnvUtils


class LLM:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None, env_prefix: str = "EBS_LLM"):
        env_model = EnvUtils.get_env(f"{env_prefix}_MODEL", EnvUtils.get_env("EBS_LLM_MODEL", ""))
        env_base_url = EnvUtils.get_env(f"{env_prefix}_BASE_URL", EnvUtils.get_env("EBS_LLM_BASE_URL", ""))
        env_api_key = EnvUtils.get_env(f"{env_prefix}_API_KEY", EnvUtils.get_env("EBS_LLM_API_KEY", ""))
        env_timeout = float(EnvUtils.get_env(f"{env_prefix}_TIMEOUT", EnvUtils.get_env("EBS_LLM_TIMEOUT", "600")))

        self.model_name = model or env_model
        resolved_base_url = base_url or env_base_url
        resolved_api_key = api_key or env_api_key
        if not self.model_name or not resolved_base_url or not resolved_api_key:
            raise ValueError(
                f"Missing model config. Need model/base_url/api_key or env vars under `{env_prefix}_*` or `EBS_LLM_*`."
            )
        self.client = openai.OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=httpx.Timeout(timeout=env_timeout, connect=min(env_timeout, 30.0)),
        )

    def chat(
        self,
        messages_or_prompt,
        max_tokens=4096,
        temperature=0,
        max_retries=3,
        return_reasoning=False,
        return_usage=False,
    ):
        last_error: Exception | None = None
        for _ in range(max_retries):
            try:
                if isinstance(messages_or_prompt, str):
                    messages = [{"role": "user", "content": messages_or_prompt}]
                elif isinstance(messages_or_prompt, list):
                    messages = messages_or_prompt
                else:
                    raise ValueError("messages_or_prompt must be a string or a list of messages.")

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                response_text = response.choices[0].message.content.strip()
                usage = response.usage.model_dump() if response.usage is not None else None

                if return_reasoning:
                    reasoning = response.choices[0].message.reasoning_content
                    if return_usage:
                        return response_text, reasoning, usage
                    return response_text, reasoning
                if return_usage:
                    return response_text, usage
                return response_text

            except Exception as e:
                last_error = e
                error = f"An unexpected error occurred: {e}"
                print(error)
            time.sleep(10)
        if last_error is not None:
            raise RuntimeError(f"LLM chat failed after {max_retries} retries: {last_error}") from last_error
        raise RuntimeError("LLM chat failed without a captured exception.")
