import pytest
from pydantic import ValidationError

from agents.recommender import Recommendation, recommend
from config import AGENT_SETTINGS
from tests.conftest import FakeClient, make_response
from tests.test_classifier import CLASSIFICATION
from tests.test_risk_analyzer import ASSESSMENT

RECOMMENDATION = Recommendation(
    overall_rating="Low",
    decision="go",
    required_controls=["Restrict access to meeting participants"],
    human_review="None beyond standard code review",
    conditions=[],
)


def test_recommend_includes_all_prior_outputs():
    client = FakeClient([make_response(RECOMMENDATION)])
    settings = AGENT_SETTINGS["recommender"]

    result = recommend(client, "Summarize team meetings", CLASSIFICATION, ASSESSMENT, settings)

    assert result == RECOMMENDATION
    content = client.calls[0]["messages"][0]["content"]
    assert "Summarize team meetings" in content
    assert "<classification>" in content and "<risk_assessment>" in content
    assert '"privacy"' in content


def test_decision_values_are_constrained():
    with pytest.raises(ValidationError):
        Recommendation(
            overall_rating="Low",
            decision="maybe",
            required_controls=[],
            human_review="x",
            conditions=[],
        )
