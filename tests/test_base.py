import pytest
from pydantic import BaseModel

from agents.base import AgentError, run_agent
from config import AgentSettings, get_profile
from tests.conftest import FakeClient, make_response

SETTINGS = AgentSettings("claude-opus-5-5", "low")


class Echo(BaseModel):
    value: str


def call(client, **overrides):
    kwargs = dict(name="echo", settings=SETTINGS, system="sys", user_content="hi", schema=Echo)
    kwargs.update(overrides)
    return run_agent(client, **kwargs)


def test_returns_parsed_output_and_sends_expected_request():
    client = FakeClient([make_response(Echo(value="ok"))])
    assert call(client) == Echo(value="ok")

    sent = client.calls[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_format"] is Echo
    assert sent["output_config"] == {"effort": "low"}
    assert sent["system"] == "sys"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_bad_stop_reason_raises(stop_reason):
    client = FakeClient([make_response(Echo(value="x"), stop_reason=stop_reason)])
    with pytest.raises(AgentError, match=stop_reason.replace("refusal", "declined")):
        call(client)


def test_missing_parsed_output_raises():
    client = FakeClient([make_response(None)])
    with pytest.raises(AgentError, match="structured output"):
        call(client)


def test_usage_is_recorded():
    log = []
    client = FakeClient([make_response(Echo(value="ok"), input_tokens=7, output_tokens=3)])
    call(client, usage_log=log)
    assert (log[0].agent, log[0].input_tokens, log[0].output_tokens) == ("echo", 7, 3)


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        get_profile("nope")
