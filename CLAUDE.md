# GenAI Use Case Risk Screener

## Purpose
Takes a short description of a proposed GenAI use case and returns a
risk rating plus recommendations. Portfolio project for learning
multi-agent orchestration. Use fictional use cases only.

## Pipeline (build sequentially first)
1. Classifier: category (e.g., internal productivity, customer-facing,
   decision support), data sensitivity, user population.
2. Risk Analyzer: risks across privacy, security, bias, accuracy,
   regulatory. Rate each Low/Medium/High with a one-line rationale.
3. Recommender: overall rating, required controls, go/no-go/conditional,
   human-review needs.

## Rules
- Python 3.11+, Anthropic SDK, API key from env var ANTHROPIC_API_KEY
- Each agent returns structured JSON validated against a schema
- Keep each agent in its own module under /agents
- CLI (`python main.py "use case text"`) and Streamlit UI (`streamlit run app.py`)
- Add tests with 3 fictional sample use cases in /samples
- Small commits, one per working step

## Out of scope for now
Databases, auth