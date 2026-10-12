"""Streamlit UI: streamlit run app.py"""

import os
from pathlib import Path

import anthropic
import streamlit as st

from agents.base import AgentError, UsageRecord
from pipeline import STEPS, ScreeningResult, run_pipeline

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
STEP_LABELS = {
    "classifier": "Classifying the use case",
    "risk_analyzer": "Analyzing risks",
    "recommender": "Writing the recommendation",
}
RATING_COLORS = {"Low": "green", "Medium": "orange", "High": "red"}
DECISION_COLORS = {"go": "green", "conditional": "orange", "no_go": "red"}
DECISION_LABELS = {"go": "Go", "conditional": "Conditional", "no_go": "No-go"}


def load_samples() -> dict[str, str]:
    return {
        p.stem.split("_", 1)[1].replace("_", " ").capitalize(): p.read_text(encoding="utf-8").strip()
        for p in sorted(SAMPLES_DIR.glob("*.txt"))
    }


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def badge(text: str, color: str) -> str:
    return f":{color}-badge[{text}]"


def render_result(result: ScreeningResult) -> None:
    c, r, rec = result.classification, result.risks, result.recommendation

    col1, col2 = st.columns(2)
    col1.markdown(f"#### Overall risk\n## {badge(rec.overall_rating, RATING_COLORS[rec.overall_rating])}")
    col2.markdown(
        f"#### Decision\n## {badge(DECISION_LABELS[rec.decision], DECISION_COLORS[rec.decision])}"
    )

    st.subheader("Classification")
    cols = st.columns(3)
    cols[0].metric("Category", c.category.replace("_", " "))
    cols[1].metric("Data sensitivity", c.data_sensitivity)
    cols[2].metric("User population", c.user_population.replace("_", " "))
    st.caption(c.rationale)

    st.subheader("Risks")
    for dim, item in r:
        left, right = st.columns([1, 5])
        left.markdown(f"**{dim.capitalize()}**  \n{badge(item.rating, RATING_COLORS[item.rating])}")
        right.write(item.rationale)

    st.subheader("Required controls")
    if rec.required_controls:
        st.markdown("\n".join(f"- {ctl}" for ctl in rec.required_controls))
    else:
        st.write("None")

    if rec.conditions:
        st.subheader("Conditions before launch")
        st.markdown("\n".join(f"- [ ] {cond}" for cond in rec.conditions))

    st.subheader("Human review")
    st.write(rec.human_review)

    st.download_button(
        "Download JSON",
        data=result.model_dump_json(indent=2),
        file_name="screening_result.json",
        mime="application/json",
    )


def run_screening(text: str) -> ScreeningResult | None:
    usage: list[UsageRecord] = []
    with st.status("Screening...", expanded=True) as status:

        def on_step(name: str) -> None:
            n = STEPS.index(name) + 1
            status.update(label=f"Step {n}/{len(STEPS)}: {STEP_LABELS[name]}...")
            st.write(f"{STEP_LABELS[name]}...")

        try:
            result = run_pipeline(text, client=get_client(), usage_log=usage, on_step=on_step)
        except AgentError as e:
            status.update(label="Screening failed", state="error")
            st.error(str(e))
            return None
        except anthropic.APIStatusError as e:
            status.update(label="Screening failed", state="error")
            st.error(f"API error {e.status_code}: {e.message}")
            return None
        except anthropic.APIConnectionError:
            status.update(label="Screening failed", state="error")
            st.error("Could not reach the Anthropic API.")
            return None

        tokens = sum(u.input_tokens + u.output_tokens for u in usage)
        status.update(label=f"Screening complete ({tokens:,} tokens)", state="complete", expanded=False)
    return result


def main() -> None:
    st.set_page_config(page_title="GenAI Risk Screener", page_icon="🛡️", layout="centered")
    st.title("GenAI Use Case Risk Screener")
    st.write(
        "Describe a proposed GenAI use case. Three agents classify it, rate its risks, "
        "and recommend a decision with required controls. Use fictional use cases only."
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("Set the ANTHROPIC_API_KEY environment variable, then restart the app.")
        st.stop()

    samples = load_samples()
    choice = st.selectbox("Start from a sample (optional)", ["(none)", *samples])
    if choice != "(none)" and st.session_state.get("loaded_sample") != choice:
        st.session_state.use_case = samples[choice]
        st.session_state.loaded_sample = choice

    text = st.text_area("Use case description", key="use_case", height=180)

    if st.button("Screen use case", type="primary", disabled=not text.strip()):
        st.session_state.result = run_screening(text.strip())

    if st.session_state.get("result"):
        st.divider()
        render_result(st.session_state.result)


main()
