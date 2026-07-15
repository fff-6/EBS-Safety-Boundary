from typing import Literal

# from openai import NOT_GIVEN
from agents import ModelSettings
from pydantic import ConfigDict, Field

from ..utils import EnvUtils
from .base_config import ConfigBaseModel


class ModelProviderConfig(ConfigBaseModel):
    """config for model provider"""

    type: Literal["chat.completions", "responses", "litellm"] = "chat.completions"
    """model type, supported types: chat.completions, responses"""
    model: str = EnvUtils.get_env("EBS_LLM_MODEL", "Qwen/Qwen3-8B")
    """model name"""
    base_url: str | None = None
    """model provider base url"""
    api_key: str | None = None
    """model provider api key"""
    timeout: float | None = None
    """Client timeout in seconds for model provider requests."""


class ModelSettingsConfig(ConfigBaseModel, ModelSettings):
    """ModelSettings in openai-agents"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelParamsConfig(ConfigBaseModel):
    """Basic params shared in chat.completions and responses"""

    temperature: float | None = None
    top_p: float | None = None
    parallel_tool_calls: bool | None = None


class ModelConfigs(ConfigBaseModel):
    """Overall model config"""

    model_provider: ModelProviderConfig = Field(default_factory=ModelProviderConfig)
    """config for model provider"""
    model_settings: ModelSettingsConfig = Field(default_factory=ModelSettingsConfig)
    """config for agent's model settings"""
    model_params: ModelParamsConfig = Field(default_factory=ModelParamsConfig)
    """config for basic model usage, e.g. `query_one` in tools / judger"""
