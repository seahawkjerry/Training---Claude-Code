from agents.base import UsageRecord
from eval.compare_models import cost_usd, summarize


def test_cost_uses_model_pricing():
    usage = [
        UsageRecord("classifier", "claude-opus-5-5", 1_000_000, 0),
        UsageRecord("recommender", "claude-sonnet-5-5", 0, 1_000_000),
    ]
    assert cost_usd(usage) == 4.00 + 10.00


def test_summarize_renders_table():
    row = {
        "sample": "01_internal_meeting_summarizer",
        "rating": "Low",
        "decision": "go",
        "error": None,
        "match": True,
        "seconds": 2.0,
        "input_tokens": 3000,
        "output_tokens": 900,
        "cost": 0.03,
        "models": ["claude-opus-5-5"],
    }
    report = summarize([{"profile": "opus-low", "rows": [row]}], repeats=1)
    assert "| opus-low | 1/1 | 1/1 |" in report
