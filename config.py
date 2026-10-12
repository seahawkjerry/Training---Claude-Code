"""Model and effort used by each agent."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSettings:
    model: str
    effort: str


AGENT_SETTINGS: dict[str, AgentSettings] = {
    "classifier": AgentSettings("claude-opus-5-5", "low"),
    "risk_analyzer": AgentSettings("claude-opus-5-5", "low"),
    "recommender": AgentSettings("claude-opus-5-5", "low"),
}
