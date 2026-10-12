import json
from pathlib import Path

import pytest

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


def test_report_contains_key_fields():
    report = format_report(run_pipeline("x", client=canned_client()))
    assert "OVERALL: LOW" in report and "DECISION: GO" in report
    assert "privacy" in report and "Human review" in report


SAMPLES = sorted((Path(__file__).resolve().parent.parent / "samples").glob("*.txt"))


def test_three_samples_exist():
    assert len(SAMPLES) == 3


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.stem)
def test_sample_flows_through_pipeline(sample):
    text = sample.read_text(encoding="utf-8")
    expected = json.loads(sample.with_suffix(".expected.json").read_text(encoding="utf-8"))
    assert set(expected) == {"overall_rating", "decision"}

    client = canned_client()
    run_pipeline(text, client=client)
    for call in client.calls:
        assert text in call["messages"][0]["content"]


def test_on_step_called_before_each_agent():
    steps = []
    run_pipeline("x", client=canned_client(), on_step=steps.append)
    assert steps == ["classifier", "risk_analyzer", "recommender"]
