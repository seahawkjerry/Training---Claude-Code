import pytest
from pydantic import ValidationError

from agents.classifier import Classification, classify
from config import get_profile
from tests.conftest import FakeClient, make_response

CLASSIFICATION = Classification(
    category="internal_productivity",
    data_sensitivity="internal",
    user_population="internal_employees",
    rationale="Summarizes internal meetings for staff.",
)


def test_classify_sends_use_case_and_returns_schema():
    client = FakeClient([make_response(CLASSIFICATION)])
    settings = get_profile("opus-low")["classifier"]

    result = classify(client, "Summarize team meetings", settings)

    assert result == CLASSIFICATION
    sent = client.calls[0]
    assert "Summarize team meetings" in sent["messages"][0]["content"]
    assert sent["output_format"] is Classification
    assert sent["model"] == settings.model


def test_schema_rejects_unknown_category():
    with pytest.raises(ValidationError):
        Classification(
            category="weapons",
            data_sensitivity="internal",
            user_population="internal_employees",
            rationale="x",
        )
