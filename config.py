"""Model/effort profiles for each agent. Select with SCREENER_PROFILE or --profile."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSettings:
    model: str
    effort: str


# Per-agent settings so a profile can mix models (e.g. cheap classifier, stronger analyst).
PROFILES: dict[str, dict[str, AgentSettings]] = {
    "opus-low": {
        "classifier": AgentSettings("claude-opus-5-5", "low"),
        "risk_analyzer": AgentSettings("claude-opus-5-5", "low"),
        "recommender": AgentSettings("claude-opus-5-5", "low"),
    },
    "sonnet-medium": {
        "classifier": AgentSettings("claude-sonnet-5-5", "medium"),
        "risk_analyzer": AgentSettings("claude-sonnet-5-5", "medium"),
        "recommender": AgentSettings("claude-sonnet-5-5", "medium"),
    },
}

DEFAULT_PROFILE = "opus-low"

# USD per million tokens (input, output), used by the model comparison eval.
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
}


def get_profile(name: str | None = None) -> dict[str, AgentSettings]:
    name = name or os.environ.get("SCREENER_PROFILE") or DEFAULT_PROFILE
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose from {sorted(PROFILES)}")
    return PROFILES[name]
