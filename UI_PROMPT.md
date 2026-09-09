# Codex prompt — Hiver Email Suggested-Response UI

> Paste everything below the line into Codex. It is self-contained.
>
> **Read this note first (for you, not Codex):** the challenge brief says *"Do NOT add web frameworks."* So the UI is built as an **isolated, optional add-on**: it lives in `ui/`, its dependencies sit behind an optional extra, and the graded path (`uv sync` → `pytest` → `run_eval.py`) never installs or imports a web framework. Do not move any of this into `src/hiver_email/`.

---

## Task

Build a small web UI for an existing Python project called **Hiver Email Suggested-Response System**. The Python engine already exists and works — **do not rewrite, refactor, or modify it.** You are adding a thin API server plus a frontend on top of it.

Repo: `https://github.com/vhmaadhav/hiver-email-suggested-response`

## What the existing system does

Given an incoming email, it retrieves the 3 most similar historical email/reply pairs from ~9,500 real Enron emails using TF-IDF, feeds them to an LLM as grounding, and returns a suggested reply. It also ships an evaluation harness that scores replies on two independent signals (semantic similarity to a reference reply + an LLM rubric judge).

## Files you MUST NOT touch

```
src/hiver_email/**      the engine - import it, never edit it
scripts/run_eval.py     evaluation runner
scripts/fetch_data.py
tests/**
pyproject.toml          EXCEPT to add the optional [ui] extra described below
README.md, FAILURES.md
results/**              generated measurement output - read-only
```

## Files you WILL create

```
ui/
├── server.py           FastAPI app: 4 endpoints, thin wrapper over the engine
├── static/
│   ├── index.html      single page, no build step
│   ├── app.js          vanilla JS - no React, no bundler, no npm
│   └── styles.css      hand-written CSS
└── README.md           how to run it
```

Add to `pyproject.toml` **only** this (so the graded path stays clean):

```toml
[project.optional-dependencies]
ui = ["fastapi>=0.110", "uvicorn[standard]>=0.29"]
```

Run command: `uv run --extra ui uvicorn ui.server:app --reload --port 8000`

## Engine API you will call

Import these exactly. Signatures are real — do not guess others.

```python
from hiver_email.data import load_pairs, split_pairs, sample_heldout
from hiver_email.retrieval import build_retriever, baseline_reply
from hiver_email.generator import SuggestedReplyGenerator
from hiver_email.judge import RubricJudge
from hiver_email.llm import LLMClient

pairs = load_pairs()                      # list[EmailPair], ~11,914
corpus, heldout = split_pairs(pairs)      # deterministic 80/20
retriever = build_retriever(corpus, kind="tfidf")   # or "dense" / "hybrid"
client = LLMClient()                      # reads OPENAI_API_KEY / OPENAI_BASE_URL / GEN_MODEL / JUDGE_MODEL from .env
generator = SuggestedReplyGenerator(retriever, client, top_k=3)
judge = RubricJudge(client)

result = generator.generate("incoming email text")
# result.reply      -> str  (empty string if generation failed)
# result.error      -> str  (non-empty means it failed - surface it, do not hide it)
# result.retrieved  -> list[RetrievedExample]
#     .example_id, .incoming_email, .human_reply, .similarity (float 0-1)

verdict = judge.judge(incoming_email, human_reply, candidate_reply, result.retrieved)
# .task_fulfillment .action_alignment .completeness .tone  -> int 1-5
# .critical_error .acceptable -> bool
# .reason -> str
# .score_100() -> float   (weighted 0-100, capped at 40 if critical_error)
```

**Loading the corpus and building the index takes ~5 seconds.** Do it ONCE at startup in a FastAPI `lifespan` handler and hold it in module state. Never rebuild per request.

`generator.generate()` is a blocking network call taking 10–30s. Define endpoints that call it with `def` (not `async def`) so FastAPI runs them in its threadpool and one slow request cannot block the event loop.

## Endpoints

### `GET /api/health`
```json
{"status":"ok","corpus_size":9531,"gen_model":"openai/gpt-oss-20b","retriever":"tfidf"}
```

### `POST /api/suggest`
Request: `{"email":"...", "top_k":3}`

Response:
```json
{
  "suggested_reply": "Sure, I'll have the revised deck to you by the end of today.",
  "baseline_reply": "Attached is a clean and redline of the revised confirm.",
  "retrieved": [
    {"example_id":"beae6883bc2b05fe","incoming_email":"...","human_reply":"...","similarity":0.3134}
  ],
  "model": "openai/gpt-oss-20b",
  "elapsed_seconds": 12.4,
  "error": ""
}
```
`baseline_reply` is `baseline_reply(result.retrieved)` — the nearest historical reply copied verbatim. **The UI must show both side by side.** This contrast is the whole point of the product: the baseline is real human text that often falsely claims an attachment, and the generated reply does not.

If `result.error` is non-empty, return HTTP 200 with the error populated and empty strings elsewhere. The frontend renders the error inline. Never fabricate a reply.

### `POST /api/judge`
Request: `{"email":"...","human_reply":"...","candidate_reply":"..."}`
Response: the six verdict fields plus `judge_score` (from `.score_100()`) and `reason`. On judge failure return HTTP 200 with `{"error":"..."}` — never a defaulted score.

### `GET /api/examples?n=8`
Returns `n` held-out examples for one-click demoing:
```json
[{"example_id":"...","incoming_email":"...","human_reply":"..."}]
```
Use `sample_heldout(heldout, n)`. These were never in the retrieval corpus, so they are honest demo inputs.

### `GET /api/metrics`
Read and return `results/metrics.json` verbatim. If absent, return `{"available":false}`. **Do not compute or invent metrics in the server.**

## Frontend

Single page, vanilla JS, no build step, no npm, no CDN frameworks. `fetch()` against the endpoints above.

**Layout — one column, max-width ~880px, centered:**

1. **Header.** "Suggested Email Response" and one line of sub-text: `Retrieval-grounded drafting over 9,531 historical emails`. Small monospace status chip on the right showing model name from `/api/health`.
2. **Input.** A `<textarea>`, 6 rows, placeholder `Paste an incoming email...`. Below it: a primary "Generate reply" button, and a subtle "Load an example" link that pulls from `/api/examples` and fills the textarea.
3. **Result — two panes side by side** (stack vertically under 720px):
   - Left, labelled **Suggested reply** — the generated text. This is the primary result: give it the stronger border and background.
   - Right, labelled **Baseline: nearest historical reply** — visually de-emphasised (muted text, dashed border) with a small caption: `Copied verbatim from the closest match — shown for comparison`.
   - A copy-to-clipboard button on the suggested reply only.
4. **Grounding.** A collapsed `<details>` element: `Grounding — 3 retrieved emails`. Expanded, each shows similarity as a right-aligned monospace number (3 decimals), the historical incoming email (clamped to ~3 lines), and the reply that was actually sent. Do not hide the similarity score — it is the honest signal of how good the match was, and on this corpus it is often only ~0.2.
5. **Evaluate this reply** (secondary action). Calls `/api/judge` and renders the four rubric dimensions as small labelled 1–5 bars, plus the judge score out of 100 and the one-line reason. If `critical_error` is true, show a clear (non-decorative) warning row: `Critical error — score capped at 40`.
6. **Footer.** Pull `/api/metrics` and render one quiet line of measured numbers, each labelled as measured, e.g. `Measured on 39 held-out examples: overall 61.6 vs baseline 11.4`. If metrics are unavailable, render nothing. **Never hardcode these numbers in the frontend** — read them from the endpoint.

## Design constraints — read carefully

The reviewer is a senior engineer. The UI should look like a considered internal tool, not a generated landing page.

**Do NOT use:**
- purple/violet/indigo, or any purple→blue gradient
- gradient text, gradient buttons, gradient backgrounds of any kind
- glassmorphism, blur, neon glows, heavy drop shadows
- emoji as UI icons, or decorative emoji anywhere
- rounded-everything (no `border-radius: 9999px` on cards)
- animated gradient borders, shimmer, pulsing, bouncing
- centered hero text with an oversized headline
- "✨ AI-powered" phrasing, or any marketing copy

**Do use:**
- A near-neutral palette: an off-white page (`#fafaf9`), near-black text (`#1c1917`), one muted grey for secondary text (`#78716c`), hairline borders (`#e7e5e4`). Exactly **one** accent colour, used sparingly for the primary button and focus rings — a restrained slate blue (`#3b5b7a`) or a deep ink. Reserve a muted amber (`#a16207`) strictly for the critical-error warning.
- System font stack: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`. Monospace (`ui-monospace, "SF Mono", Menlo, monospace`) only for similarity scores, model names and metric numbers.
- Generous whitespace, left-aligned text, `border-radius: 4px` at most.
- Real states: a disabled button with a plain "Generating…" label while a request is in flight (generation takes 10–30s, so this matters), an inline error row, and an empty state.
- Full dark-mode support via `@media (prefers-color-scheme: dark)` using the same restraint.
- Keyboard: Cmd/Ctrl+Enter submits. Visible focus rings. Real `<label>` elements.

Think Linear, Stripe docs, or a well-made internal admin tool. Quiet, dense, typographic.

## Correctness requirements

- Never invent a reply, a score, or a metric in the frontend. Every number rendered must come from an API response.
- Surface errors as errors. A failed generation shows the error text; it does not show a blank success state.
- Escape all text before inserting it into the DOM — this content is real email text with arbitrary characters. Use `textContent`, not `innerHTML`.
- Handle the slow path: a 30s request must not look frozen.
- `ui/README.md` documents the run command and states that the UI requires `.env` to be configured, and that it is an optional add-on outside the graded evaluation path.

## Definition of done

- `uv run --extra ui uvicorn ui.server:app --port 8000` serves the page at `http://localhost:8000`.
- Typing an email and pressing Generate returns a suggested reply, the baseline, and 3 grounding examples with similarity scores.
- "Evaluate this reply" returns rubric scores.
- `uv run pytest` still passes untouched, and `uv sync` without `--extra ui` still installs no web framework.
