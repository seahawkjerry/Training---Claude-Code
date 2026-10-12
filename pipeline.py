"""Sequential orchestration: Classifier -> Risk Analyzer -> Recommender."""

from collections.abc import Callable

from pydantic import BaseModel

from agents.base import UsageRecord
from agents.classifier import Classification, classify
from agents.recommender import Recommendation, recommend
from agents.risk_analyzer import RiskAssessment, analyze
from config import AGENT_SETTINGS

STEPS = ["classifier", "risk_analyzer", "recommender"]


class ScreeningResult(BaseModel):
    use_case: str
    classification: Classification
    risks: RiskAssessment
    recommendation: Recommendation


def run_pipeline(
    use_case: str,
    client=None,
    usage_log: list[UsageRecord] | None = None,
    on_step: Callable[[str], None] | None = None,
) -> ScreeningResult:
    """Run the three agents in order. on_step(name) is called before each agent starts."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    settings = AGENT_SETTINGS
    notify = on_step or (lambda _name: None)

    notify("classifier")
    classification = classify(client, use_case, settings["classifier"], usage_log)
    notify("risk_analyzer")
    risks = analyze(client, use_case, classification, settings["risk_analyzer"], usage_log)
    notify("recommender")
    recommendation = recommend(
        client, use_case, classification, risks, settings["recommender"], usage_log
    )
    return ScreeningResult(
        use_case=use_case,
        classification=classification,
        risks=risks,
        recommendation=recommendation,
    )
