# ruff: noqa


def _bootstrap_ebs() -> None:
    try:
        from agents.run import set_default_agent_runner
    except ModuleNotFoundError:
        return

    from .utils import EnvUtils, setup_logging
    from .patch.runner import EBSAgentRunner
    from .tracing import setup_tracing

    if not (__import__("os").getenv("EBS_LLM_TYPE") and __import__("os").getenv("EBS_LLM_MODEL")):
        return

    setup_logging(EnvUtils.get_env("EBS_LOG_LEVEL", "WARNING"))
    setup_tracing()
    set_default_agent_runner(EBSAgentRunner())


_bootstrap_ebs()
