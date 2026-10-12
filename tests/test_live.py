"""Real-API end-to-end run over the samples. Run with: pytest -m live"""

import json
from pathlib import Path

import pytest

from pipeline import run_pipeline

SAMPLES = sorted((Path(__file__).resolve().parent.parent / "samples").glob("*.txt"))


@pytest.mark.live
@pytest.mark.parametrize("sample", SAMPLES, ids=lambda p: p.stem)
def test_sample_matches_expected(sample, require_api_key):
    expected = json.loads(sample.with_suffix(".expected.json").read_text(encoding="utf-8"))

    result = run_pipeline(sample.read_text(encoding="utf-8"))

    rec = result.recommendation
    assert rec.overall_rating in expected["overall_rating"]
    assert rec.decision in expected["decision"]
    if rec.decision == "conditional":
        assert rec.conditions, "conditional decision should list conditions"
