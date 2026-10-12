"""Agent 2: rate risks across five dimensions, informed by the classification."""

from typing import Literal

from pydantic import BaseModel, Field

from agents.base import UsageRecord, run_agent
from agents.classifier import Classification
from config import AgentSettings

Rating = Literal["Low", "Medium", "High"]

SYSTEM_PROMPT = """\
You are the risk analyst in a GenAI use-case risk review. You receive a use-case description \
and the intake classification. Rate the use case on five dimensions, each Low, Medium, or High, \
with a one-line rationale tied to specifics of this use case (not generic boilerplate).

- privacy: exposure, retention, or misuse of personal data.
- security: prompt injection, data exfiltration, unauthorized actions, access control.
- bias: unfair or discriminatory outcomes for groups of people.
- accuracy: harm caused by hallucinated, outdated, or wrong outputs.
- regulatory: obligations under laws or sector rules (e.g. data protection, consumer \
protection, lending, employment, health, AI-specific regulation).

Calibration: High means likely, material harm without strong controls. Medium means plausible \
harm that standard controls can manage. Low means limited harm even if something goes wrong. \
Weigh the classification: regulated data, decisions about people, and public-facing outputs \
generally raise ratings; internal, low-stakes drafting tools generally lower them."""


class RiskItem(BaseModel):
    rating: Rating
    rationale: str = Field(description="One line, specific to this use case.")


class RiskAssessment(BaseModel):
    privacy: RiskItem
    security: RiskItem
    bias: RiskItem
    accuracy: RiskItem
    regulatory: RiskItem


def analyze(
    client,
    use_case: str,
    classification: Classification,
    settings: AgentSettings,
    usage_log: list[UsageRecord] | None = None,
) -> RiskAssessment:
    user_content = (
        f"<use_case>\n{use_case}\n</use_case>\n\n"
        f"<classification>\n{classification.model_dump_json(indent=2)}\n</classification>"
    )
    return run_agent(
        client,
        name="risk_analyzer",
        settings=settings,
        system=SYSTEM_PROMPT,
        user_content=user_content,
        schema=RiskAssessment,
        usage_log=usage_log,
    )
