"""Agent 3: turn the classification and risk ratings into a decision and required controls."""

from typing import Literal

from pydantic import BaseModel, Field

from agents.base import UsageRecord, run_agent
from agents.classifier import Classification
from agents.risk_analyzer import RiskAssessment
from config import AgentSettings

SYSTEM_PROMPT = """\
You are the final reviewer in a GenAI use-case risk review. You receive the use-case \
description, its classification, and per-dimension risk ratings. Produce the recommendation.

- overall_rating: Low, Medium, or High. It is usually the highest individual rating, unless \
you explain why that risk is narrow enough not to dominate.
- decision: go (proceed with standard practices), conditional (proceed only once the listed \
conditions are met), or no_go (do not proceed as described).
- required_controls: concrete controls matched to the risks rated Medium or High, e.g. \
"Redact customer PII before sending transcripts to the model". Avoid vague items like \
"ensure compliance".
- human_review: who must review what, and when (before launch, per output, periodic audit). \
Say "None beyond standard code review" if that is genuinely enough.
- conditions: the specific conditions that must be met before launch. Required when the \
decision is conditional; empty list otherwise."""


class Recommendation(BaseModel):
    overall_rating: Literal["Low", "Medium", "High"]
    decision: Literal["go", "conditional", "no_go"]
    required_controls: list[str]
    human_review: str = Field(description="Who reviews what, and when.")
    conditions: list[str] = Field(description="Conditions to meet before launch; empty unless conditional.")


def recommend(
    client,
    use_case: str,
    classification: Classification,
    risks: RiskAssessment,
    settings: AgentSettings,
    usage_log: list[UsageRecord] | None = None,
) -> Recommendation:
    user_content = (
        f"<use_case>\n{use_case}\n</use_case>\n\n"
        f"<classification>\n{classification.model_dump_json(indent=2)}\n</classification>\n\n"
        f"<risk_assessment>\n{risks.model_dump_json(indent=2)}\n</risk_assessment>"
    )
    return run_agent(
        client,
        name="recommender",
        settings=settings,
        system=SYSTEM_PROMPT,
        user_content=user_content,
        schema=Recommendation,
        usage_log=usage_log,
    )
