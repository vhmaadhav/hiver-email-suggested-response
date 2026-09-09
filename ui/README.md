# Hiver suggested-response UI

This directory is an optional web interface over the existing Hiver email engine. It is outside the graded evaluation path and does not modify the engine, evaluation runner, tests, or committed measurements.

## Run

Configure the repository's `.env` first. The UI uses the same `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GEN_MODEL`, and `JUDGE_MODEL` settings as the existing engine. The Enron dataset must also be present under `data/`; use the repository's existing data-fetching instructions if needed.

From the repository root, run:

```console
uv run --extra ui uvicorn ui.server:app --reload --port 8000
```

Then open <http://localhost:8000>.

The corpus and TF-IDF index are loaded once when the server starts. Generating and evaluating replies make blocking model calls, which the synchronous FastAPI endpoints run in worker threads.
