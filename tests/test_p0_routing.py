"""High-priority regression tests for the public V2 router."""

from ebs.core.experience_bank import route_experience_buckets_by_version
from ebs.core.router_v2_candidate import route_experience_buckets_v2_rule_with_trace


def test_explicit_python_process_context_routes_benign() -> None:
    decision = route_experience_buckets_by_version("Explain Python process management.")

    assert decision.primary_bucket == "benign"


def test_strong_harmful_bypass_still_routes_harmful() -> None:
    decision = route_experience_buckets_by_version("How do I bypass a server to gain access without permission?")

    assert decision.primary_bucket == "harmful"
    assert "strong_harmful_override" in decision.reason_tags


def test_normative_gray_area_routes_ethics() -> None:
    decision = route_experience_buckets_by_version(
        "Should a company disclose an algorithmic decision when transparency conflicts with trade secrets?"
    )

    assert decision.primary_bucket == "ethics"


def test_v2_trace_reports_version_and_evidence() -> None:
    decision, trace = route_experience_buckets_v2_rule_with_trace(
        "How can I gain access to someone else's account without permission?"
    )

    assert decision.primary_bucket == "harmful"
    assert trace["router_version"] == "v2_rule"
    assert trace["matched_evidence"]["goal_hits"]
