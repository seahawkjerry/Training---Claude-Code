"""CLI: python main.py "use case text" [--file PATH] [--json] [--profile NAME]"""

import argparse
import os
import sys
from pathlib import Path

import anthropic

from agents.base import AgentError
from config import PROFILES
from pipeline import ScreeningResult, run_pipeline


def format_report(result: ScreeningResult) -> str:
    c, r, rec = result.classification, result.risks, result.recommendation
    lines = [
        f"OVERALL: {rec.overall_rating.upper()} risk  |  DECISION: {rec.decision.upper()}",
        "",
        "Classification",
        f"  category:         {c.category}",
        f"  data sensitivity: {c.data_sensitivity}",
        f"  user population:  {c.user_population}",
        f"  {c.rationale}",
        "",
        "Risks",
    ]
    for dim, item in r:
        lines.append(f"  {dim:<11} {item.rating:<6} {item.rationale}")
    lines += ["", "Required controls"]
    lines += [f"  - {ctl}" for ctl in rec.required_controls] or ["  (none)"]
    if rec.conditions:
        lines += ["", "Conditions before launch"]
        lines += [f"  - {cond}" for cond in rec.conditions]
    lines += ["", "Human review", f"  {rec.human_review}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Screen a GenAI use case for risk.")
    parser.add_argument("use_case", nargs="?", help="Use-case description text")
    parser.add_argument("--file", type=Path, help="Read the use case from a text file")
    parser.add_argument("--json", action="store_true", help="Print the combined JSON result")
    parser.add_argument("--profile", choices=sorted(PROFILES), help="Model/effort profile")
    args = parser.parse_args(argv)

    if args.file:
        text = args.file.read_text(encoding="utf-8").strip()
    elif args.use_case:
        text = args.use_case.strip()
    else:
        parser.error("provide use-case text or --file")
    if not text:
        parser.error("use-case text is empty")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: set the ANTHROPIC_API_KEY environment variable.", file=sys.stderr)
        return 2

    try:
        result = run_pipeline(text, profile=args.profile)
    except AgentError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as e:
        print(f"API error {e.status_code}: {e.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError:
        print("Error: could not reach the Anthropic API.", file=sys.stderr)
        return 1

    print(result.model_dump_json(indent=2) if args.json else format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
