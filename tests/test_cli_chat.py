from ebs.runtime.ui.cli_chat import normalize_agent_config_name


def test_normalize_agent_config_name_accepts_short_yaml_path():
    assert normalize_agent_config_name("simple/base.yaml") == "simple/base"


def test_normalize_agent_config_name_accepts_full_repo_path():
    assert normalize_agent_config_name("configs/agents/simple/base.yaml") == "agents/simple/base"


def test_normalize_agent_config_name_normalizes_windows_separators():
    assert normalize_agent_config_name(r"configs\agents\simple\base.yaml") == "agents/simple/base"
