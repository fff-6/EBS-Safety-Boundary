"""Local vLLM compatibility wrapper used by repository evaluation scripts."""

from __future__ import annotations

from typing import Any

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class vLLM:
    """Small adapter matching the subset of the upstream RedEval interface we use."""

    def __init__(self, model_kwargs: dict[str, Any]):
        self.model_kwargs = model_kwargs
        self.llm = LLM(**model_kwargs)

    def get_name(self) -> str:
        return str(self.model_kwargs["model"]).split("/")[-1]

    def generate(self, query: str, sampling_params: dict[str, Any]) -> str:
        outputs = self.llm.generate([query], SamplingParams(**sampling_params))
        return outputs[0].outputs[0].text

    def batch_generate(self, queries: list[str], sampling_params: dict[str, Any]) -> list[str]:
        outputs = self.llm.generate(queries, SamplingParams(**sampling_params))
        return [output.outputs[0].text for output in outputs]

    def generate_format(
        self,
        query: str,
        sampling_params: dict[str, Any],
        response_format: Any,
    ) -> str:
        schema = response_format.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=schema)
        outputs = self.llm.generate(
            [query],
            SamplingParams(**sampling_params, guided_decoding_params=guided_decoding_params),
        )
        return outputs[0].outputs[0].text

    def batch_generate_format(
        self,
        queries: list[str],
        sampling_params: dict[str, Any],
        response_format: Any,
    ) -> list[str]:
        schema = response_format.model_json_schema()
        guided_decoding_params = GuidedDecodingParams(json=schema)
        outputs = self.llm.generate(
            queries,
            SamplingParams(**sampling_params, guided_decoding_params=guided_decoding_params),
        )
        return [output.outputs[0].text for output in outputs]
