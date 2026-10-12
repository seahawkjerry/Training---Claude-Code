# GenAI Use Case Risk Screener

Sequential 3-agent pipeline (Classifier -> Risk Analyzer -> Recommender) that screens a
short, fictional GenAI use-case description and returns a risk rating and recommendations.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
export ANTHROPIC_API_KEY=...                               # or set in your shell
```

## Usage
```bash
python main.py "An internal tool that summarizes meeting transcripts for employees"
python main.py --file samples/02_customer_support_chatbot.txt --json
```

## Tests
```bash
pytest            # offline, uses a fake client
pytest -m live    # real API, costs a few cents
```
