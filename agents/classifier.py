"""Agent 1: classify a proposed GenAI use case."""

from typing import Literal

from pydantic import BaseModel, Field

from agents.base import UsageRecord, run_agent
from config import AgentSettings

SYSTEM_PROMPT = """\
You are the intake classifier in a GenAI use-case risk review. You receive a short \
description of a proposed GenAI use case and classify it so later reviewers can assess risk.

Classify three things:
- category: the primary purpose. internal_productivity (helps employees do internal work), \
customer_facing (outputs reach customers directly), decision_support (informs or makes \
decisions about people, money, health, or access), content_generation (produces marketing, \
creative, or published content), or other.
- data_sensitivity: the most sensitive data the system will touch. public, internal, \
confidential (business secrets, non-regulated personal data), or regulated (data covered \
by laws such as health, financial, children's, or biometric data).
- user_population: who interacts with or is affected by the system's outputs.

If the description is ambiguous, pick the more cautious option and say why in the rationale. \
Keep the rationale to two or three sentences."""


class Classification(BaseModel):
    category: Literal[
        "internal_productivity",
        "customer_facing",
        "decision_support",
        "content_generation",
        "other",
    ]
    data_sensitivity: Literal["public", "internal", "confidential", "regulated"]
    user_population: Literal["internal_employees", "external_customers", "general_public", "mixed"]
    rationale: str = Field(description="Two or three sentences explaining the classification.")


def classify(
    client,
    use_case: str,
    settings: AgentSettings,
    usage_log: list[UsageRecord] | None = None,
) -> Classification:
    return run_agent(
        client,
        name="classifier",
        settings=settings,
        system=SYSTEM_PROMPT,
        user_content=f"<use_case>\n{use_case}\n</use_case>",
        schema=Classification,
        usage_log=usage_log,
    )
