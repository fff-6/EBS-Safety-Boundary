from __future__ import annotations

from typing import Any

try:
    from .base import AsyncBaseToolkit as AsyncBaseToolkit
except ModuleNotFoundError:
    class AsyncBaseToolkit:  # type: ignore[no-redef]
        """Fallback placeholder when agent dependencies are unavailable."""


try:
    from .utils import get_tools_map as get_tools_map, get_tools_schema as get_tools_schema, register_tool as register_tool
except ModuleNotFoundError:
    def _missing_agents_dependency(*args: Any, **kwargs: Any) -> Any:
        raise ModuleNotFoundError("Optional dependency `agents` is required for toolkit registration helpers.")

    get_tools_map = _missing_agents_dependency  # type: ignore[assignment]
    get_tools_schema = _missing_agents_dependency  # type: ignore[assignment]
    register_tool = _missing_agents_dependency  # type: ignore[assignment]


TOOLKIT_MAP: dict[str, type[Any]] = {}


def _register_optional_toolkit(name: str, module_name: str, class_name: str) -> None:
    def _lazy_toolkit_factory(*args: Any, **kwargs: Any) -> Any:
        module = __import__(f"{__name__}.{module_name}", fromlist=[class_name])
        toolkit_class = getattr(module, class_name)
        return toolkit_class(*args, **kwargs)

    TOOLKIT_MAP[name] = _lazy_toolkit_factory  # type: ignore[assignment]


_register_optional_toolkit("search", "search_toolkit", "SearchToolkit")
_register_optional_toolkit("document", "document_toolkit", "DocumentToolkit")
_register_optional_toolkit("image", "image_toolkit", "ImageToolkit")
_register_optional_toolkit("file_edit", "file_edit_toolkit", "FileEditToolkit")
_register_optional_toolkit("github", "github_toolkit", "GitHubToolkit")
_register_optional_toolkit("arxiv", "arxiv_toolkit", "ArxivToolkit")
_register_optional_toolkit("wikipedia", "wikipedia_toolkit", "WikipediaSearchTool")
_register_optional_toolkit("codesnip", "codesnip_toolkit", "CodesnipToolkit")
_register_optional_toolkit("bash", "bash_toolkit", "BashToolkit")
_register_optional_toolkit("python_executor", "python_executor_toolkit", "PythonExecutorToolkit")
_register_optional_toolkit("video", "video_toolkit", "VideoToolkit")
_register_optional_toolkit("audio", "audio_toolkit", "AudioToolkit")
_register_optional_toolkit("serper", "serper_toolkit", "SerperToolkit")
_register_optional_toolkit("tabular", "tabular_data_toolkit", "TabularDataToolkit")
_register_optional_toolkit("memory_simple", "memory_toolkit", "SimpleMemoryToolkit")
_register_optional_toolkit("user_interaction", "user_interaction_toolkit", "UserInteractionToolkit")


def get_toolkits_map(names: list[str] | None = None) -> dict[str, Any]:
    """Get all the toolkits specified by names."""

    from ..config import ConfigLoader

    toolkits: dict[str, Any] = {}
    if names is None:
        names = list(TOOLKIT_MAP.keys())
    else:
        assert all(name in TOOLKIT_MAP for name in names), f"Error config tools: {names}"
    for name in names:
        config = ConfigLoader.load_toolkit_config(name)
        toolkits[name] = TOOLKIT_MAP[name](config=config)
    return toolkits
