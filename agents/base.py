"""Shared call path for every agent: one structured-output request, validated by Pydantic."""

from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel

from config import AgentSettings

T = TypeVar("T", bound=BaseModel)

MAX_TOKENS = 4000
# Server-side fallback: if a safety classifier declines, the API retries on a suitable model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AgentError(RuntimeError):
    """The model did not return a usable structured result."""


@dataclass
class UsageRecord:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int


def run_agent(
    client,
    *,
    name: str,
    settings: AgentSettings,
    system: str,
    user_content: str,
    schema: type[T],
    usage_log: list[UsageRecord] | None = None,
) -> T:
    response = client.beta.messages.parse(
        model=settings.model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        output_format=schema,
        output_config={"effort": settings.effort},
        fallbacks="default",
        betas=[FALLBACK_BETA],
    )

    if usage_log is not None:
        usage_log.append(
            UsageRecord(
                agent=name,
                model=getattr(response, "model", settings.model),
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        )

    if response.stop_reason == "refusal":
        raise AgentError(f"{name}: model declined the request")
    if response.stop_reason == "max_tokens":
        raise AgentError(f"{name}: output hit max_tokens ({MAX_TOKENS}) before completing")

    parsed = response.parsed_output
    if parsed is None:
        raise AgentError(f"{name}: response did not contain structured output")
    return parsed
