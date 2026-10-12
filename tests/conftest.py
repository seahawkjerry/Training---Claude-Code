import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_response(parsed, stop_reason="end_turn", input_tokens=100, output_tokens=50):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        model="fake-model",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeClient:
    """Mimics client.beta.messages.parse; returns queued responses and records each call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeClient ran out of queued responses")
        return self._responses.pop(0)


@pytest.fixture
def fake_client():
    return FakeClient


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m") and "live" in config.getoption("-m"):
        return
    skip_live = pytest.mark.skip(reason="live API test; run with -m live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls the real Anthropic API (costs money)")


@pytest.fixture
def require_api_key():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
