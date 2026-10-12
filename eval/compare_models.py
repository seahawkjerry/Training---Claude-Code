"""Compare model/effort profiles on the samples: accuracy vs expected, stability, cost, latency.

Usage: python eval/compare_models.py [--repeats 2] [--profiles opus-low sonnet-medium]
Spends real API money (roughly $1-2 at the defaults). Writes eval/results.md.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402

from agents.base import AgentError, UsageRecord  # noqa: E402
from config import PRICING, PROFILES  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

SAMPLES = sorted((ROOT / "samples").glob("*.txt"))


def cost_usd(usage: list[UsageRecord]) -> float:
    total = 0.0
    for u in usage:
        # A server-side fallback may report a different model; price unknown models at Opus rates.
        price_in, price_out = PRICING.get(u.model, PRICING["claude-opus-5-5"])
        total += u.input_tokens * price_in / 1e6 + u.output_tokens * price_out / 1e6
    return total


def run_profile(client, profile: str, repeats: int) -> dict:
    rows = []
    for sample in SAMPLES:
        text = sample.read_text(encoding="utf-8")
        expected = json.loads(sample.with_suffix(".expected.json").read_text(encoding="utf-8"))
        for _ in range(repeats):
            usage: list[UsageRecord] = []
            start = time.perf_counter()
            try:
                result = run_pipeline(text, client=client, profile=profile, usage_log=usage)
                rating, decision, error = (
                    result.recommendation.overall_rating,
                    result.recommendation.decision,
                    None,
                )
            except AgentError as e:
                rating, decision, error = None, None, str(e)
            rows.append(
                {
                    "sample": sample.stem,
                    "rating": rating,
                    "decision": decision,
                    "error": error,
                    "match": rating in expected["overall_rating"]
                    and decision in expected["decision"],
                    "seconds": time.perf_counter() - start,
                    "input_tokens": sum(u.input_tokens for u in usage),
                    "output_tokens": sum(u.output_tokens for u in usage),
                    "cost": cost_usd(usage),
                    "models": sorted({u.model for u in usage}),
                }
            )
            print(f"  {profile:<14} {sample.stem:<32} {rating}/{decision} "
                  f"${rows[-1]['cost']:.4f} {rows[-1]['seconds']:.1f}s", flush=True)
    return {"profile": profile, "rows": rows}


def summarize(results: list[dict], repeats: int) -> str:
    out = [
        "# Model comparison",
        "",
        f"{len(SAMPLES)} samples x {repeats} repeats per profile. "
        "Output tokens include thinking tokens.",
        "",
        "| Profile | Valid | Matches expected | Stable | Avg $ / screening | Avg in tok | Avg out tok | Avg latency (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for res in results:
        rows = res["rows"]
        n = len(rows)
        valid = sum(r["error"] is None for r in rows)
        matches = sum(r["match"] for r in rows)
        stable = sum(
            len({(r["rating"], r["decision"]) for r in rows if r["sample"] == s.stem}) == 1
            for s in SAMPLES
        )
        out.append(
            f"| {res['profile']} | {valid}/{n} | {matches}/{n} | {stable}/{len(SAMPLES)} samples "
            f"| ${sum(r['cost'] for r in rows) / n:.4f} "
            f"| {sum(r['input_tokens'] for r in rows) / n:.0f} "
            f"| {sum(r['output_tokens'] for r in rows) / n:.0f} "
            f"| {sum(r['seconds'] for r in rows) / n:.1f} |"
        )
    out += ["", "## Per-sample outcomes", ""]
    for res in results:
        out.append(f"**{res['profile']}**")
        for s in SAMPLES:
            outcomes = Counter(
                f"{r['rating']}/{r['decision']}" if r["error"] is None else "ERROR"
                for r in res["rows"]
                if r["sample"] == s.stem
            )
            out.append(f"- {s.stem}: " + ", ".join(f"{k} x{v}" for k, v in outcomes.items()))
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    args = parser.parse_args()

    client = anthropic.Anthropic()
    results = [run_profile(client, p, args.repeats) for p in args.profiles]
    report = summarize(results, args.repeats)
    (ROOT / "eval" / "results.md").write_text(report + "\n", encoding="utf-8")
    (ROOT / "eval" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
