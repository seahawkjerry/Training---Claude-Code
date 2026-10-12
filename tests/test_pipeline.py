from agents.classifier import Classification
from agents.recommender import Recommendation
from agents.risk_analyzer import RiskAssessment
from main import format_report
from pipeline import run_pipeline
from tests.conftest import FakeClient, make_response
from tests.test_classifier import CLASSIFICATION
from tests.test_recommender import RECOMMENDATION
from tests.test_risk_analyzer import ASSESSMENT


def canned_client():
    return FakeClient(
        [make_response(CLASSIFICATION), make_response(ASSESSMENT), make_response(RECOMMENDATION)]
    )


def test_agents_run_in_order_and_hand_off_outputs():
    client = canned_client()
    log = []

    result = run_pipeline("Summarize team meetings", client=client, usage_log=log)

    schemas = [call["output_format"] for call in client.calls]
    assert schemas == [Classification, RiskAssessment, Recommendation]
    assert "<classification>" in client.calls[1]["messages"][0]["content"]
    assert "<risk_assessment>" in client.calls[2]["messages"][0]["content"]
    assert [u.agent for u in log] == ["classifier", "risk_analyzer", "recommender"]
    assert result.recommendation.decision == "go"


def test_profile_selects_models():
    client = canned_client()
    run_pipeline("x", client=client, profile="sonnet-medium")
    assert {c["model"] for c in client.calls} == {"claude-sonnet-5-5"}
    assert {c["output_config"]["effort"] for c in client.calls} == {"medium"}


def test_report_contains_key_fields():
    report = format_report(run_pipeline("x", client=canned_client()))
    assert "OVERALL: LOW" in report and "DECISION: GO" in report
    assert "privacy" in report and "Human review" in report
