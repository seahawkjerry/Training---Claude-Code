import pytest
from pydantic import ValidationError

from agents.risk_analyzer import RiskAssessment, RiskItem, analyze
from config import AGENT_SETTINGS
from tests.conftest import FakeClient, make_response
from tests.test_classifier import CLASSIFICATION

LOW = RiskItem(rating="Low", rationale="Limited exposure.")
ASSESSMENT = RiskAssessment(privacy=LOW, security=LOW, bias=LOW, accuracy=LOW, regulatory=LOW)


def test_analyze_includes_use_case_and_classification():
    client = FakeClient([make_response(ASSESSMENT)])
    settings = AGENT_SETTINGS["risk_analyzer"]

    result = analyze(client, "Summarize team meetings", CLASSIFICATION, settings)

    assert result == ASSESSMENT
    content = client.calls[0]["messages"][0]["content"]
    assert "Summarize team meetings" in content
    assert '"category": "internal_productivity"' in content
    assert client.calls[0]["output_format"] is RiskAssessment


def test_rating_must_be_low_medium_high():
    with pytest.raises(ValidationError):
        RiskItem(rating="Severe", rationale="x")


def test_all_five_dimensions_required():
    with pytest.raises(ValidationError):
        RiskAssessment(privacy=LOW, security=LOW, bias=LOW, accuracy=LOW)
