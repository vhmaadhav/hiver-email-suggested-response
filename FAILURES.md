# Failure log

A running record of what broke during the build, why, and what we changed in
response. Kept because the decisions in this repo only make sense alongside the
constraints that forced them — and because "it worked first try" is rarely true.

Entries are in the order they were hit.

---

## 1. Package build failed before any code ran

**Symptom.** `uv sync` failed with `OSError: Readme file does not exist: README.md`.

**Cause.** `pyproject.toml` declares `readme = "README.md"` and hatchling resolves
it at build time. The repo was scaffolded pyproject-first, so the file did not
exist yet.

**Fix.** Created a placeholder `README.md` immediately and wrote the real one at
the end. No design change.

---

## 2. Torch would have pulled a CUDA runtime onto a CPU-only, 16 GB machine

**Symptom.** Anticipated, not observed: `sentence-transformers` depends on
`torch`, and the default wheel on some platforms ships a multi-GB CUDA runtime.

**Why it matters here.** The target environment is 16 GB RAM, CPU only. A CUDA
build costs a long download and disk for code that can never execute.

**Fix.** Pinned the CPU wheel explicitly in `pyproject.toml`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

Verified: `torch 2.14.0+cpu`, `torch.version.cuda is None`.

---

## 3. First LLM provider: model deprecated mid-build

**Symptom.** `404 - This model models/gemini-2.5-flash is no longer available to
new users.`

**Fix.** Moved to the successor model the error named. This is exactly why
`GEN_MODEL` / `JUDGE_MODEL` are environment variables rather than constants — the
swap was a one-line `.env` edit, no code change.

---

## 4. Reasoning models returned empty completions

**Symptom.** `LLMError: model returned empty content` on a request with
`max_tokens=20`. The same prompt at `max_tokens=800` returned `"OK"` — with
`completion_tokens=1` but `total_tokens=74`.

**Cause.** The model spent the entire output budget on hidden reasoning tokens
and had nothing left for the visible message.

**Fix.** Two changes in `src/hiver_email/llm.py`:
- raised default ceilings (generation and judging both run at 1200 tokens);
- on an empty completion, double `max_tokens` (capped at 4096) and retry rather
  than failing.

This is a real class of bug, not a one-off: any reasoning-capable model can hit
it, so the retry stays in.

---

## 5. Provider switched to the NVIDIA build API mid-build

**Cause.** An `NVIDIA_BUILD-API-KEY` was supplied for the run.

**Fix.** No code change — only `.env`. The client is the OpenAI SDK pointed at
`https://integrate.api.nvidia.com/v1` via `OPENAI_BASE_URL`. This is the payoff
of writing one provider integration against an OpenAI-compatible surface instead
of a multi-provider abstraction layer.

---

## 6. Most catalogue models were not actually callable

**Symptom.** The endpoint lists 80 models. Probing 12 plausible instruct models
concurrently:

| Result | Models |
|---|---|
| worked | `openai/gpt-oss-20b`, `nvidia/nemotron-3.5-lightning-30b-a3b` |
| `404 Not Found for account` | `nemotron-nano-3-30b`, `mistral-large-2-instruct`, `kimi-k2.6`, `phi-3.5-moe`, `llama-3.1-nemotron-51b`, `granite-3.0-8b` |
| `503 Service temporarily overloaded` | `nemotron-3-super-120b-a12b` |
| timeout (25 s) | `gemma-4-31b-it`, `deepseek-v4-flash`, `mistral-nemotron` |

**Lesson.** `models.list()` is a catalogue, not an entitlement list. The first
probe ran without a per-request timeout and hung long enough to need killing;
the second passed `timeout=25.0, max_retries=0` and finished in seconds.

**Consequence.** The choice of models below was forced by availability, not
preference.

---

## 7. The generator leaked its chain-of-thought into the reply

**Symptom.** With `nvidia/nemotron-3.5-lightning-30b-a3b` generating, a
suggested reply came back as:

> `Here's a thinking process: 1. **Analyze the User's Request:** - I need to draft a suggested email reply...`

**Why this was nearly invisible.** The pipeline did not crash. Without a check,
that scratchpad would have been written into `per_response.csv` as a legitimate
suggested reply.

**What caught it.** The evaluator. The judge scored it
`task_fulfillment=1, critical_error=true, judge_score=0.0` with the reason "The
candidate reply is a meta-analysis, not an actual response". The evaluation
harness caught a generation bug before a human did — which is the argument for
building the evaluator carefully.

**Fix, two parts.**
1. Swapped roles so `openai/gpt-oss-20b`, which returns clean prose, generates.
2. Added `_strip_thinking()` in `generator.py`: keep text after `</think>`, and
   if the reply still opens with a scratchpad marker, return an empty string.
   An empty reply is recorded as an explicit failure row and excluded from the
   means. Shipping the scratchpad would have quietly polluted the scores;
   recording a failure is honest.

---

## 8. The judge could not emit JSON

**Symptom.** With `nemotron-3.5-lightning` judging:
`ValueError: no JSON object found in judge response: 'Here's a thinking process: ...'`
— it burned the whole budget reasoning and never reached the JSON.

**Fix.** `openai/gpt-oss-20b` judges. It honours the structured-output contract
and returned well-formed verdicts in testing.

**Deliberate non-fix.** `parse_json_object()` still *raises* on malformed output.
It tolerates markdown fences and leading prose, but it never substitutes a
default verdict — a silently defaulted score would fabricate a measurement. The
row is recorded with an error and dropped from the means, with `main_n_failed`
reported alongside. Tested in `tests/test_scoring.py`.

---

## 9. Generation and judging ended up on the same model

**Consequence of #7 and #8:** only two models were reachable, one of which could
neither generate clean prose nor emit JSON. So `openai/gpt-oss-20b` does both
jobs.

**This is a real limitation, not a design choice.** Self-evaluation invites
self-preference bias. It is disclosed in the README, recorded in
`metrics.json` as `config.same_model_for_gen_and_judge`, and a separate judge is
the first thing to change in a production evaluation.

---

## 10. Windows console crashed on Enron text

**Symptom.** `UnicodeEncodeError: 'charmap' codec can't encode character '‑'`
— printing a generated reply containing a non-breaking hyphen to a cp1252 console.

**Fix.** Both scripts reconfigure `stdout`/`stderr` to UTF-8 with
`errors="replace"`. CSV output was already written with `encoding="utf-8"`.

---

## 11. Sequential evaluation did not fit the time budget

**Symptom.** Measured latency: ~12 s per generation, ~7-30 s per judgement. At
40 examples × (1 generation + 2 judgements) that is ~20-30 minutes serial.

**Fix.** A `ThreadPoolExecutor` (`--workers`, default 6) over examples. Only
network waits overlap — retrieval, scoring and the split stay deterministic, and
results are re-sorted into `eval_set` order before saving so thread completion
order never leaks into the report.

---

## Things deliberately not done

- **No FAISS / vector DB / LangChain / LlamaIndex.** 9.5 k documents is a single
  sparse matrix and a dot product. TF-IDF indexes in ~0.8 s and is auditable by
  reading the code.
- **No second provider integration.** `OPENAI_BASE_URL` already covers it, as #5
  demonstrated under time pressure.
- **No fabricated human labels.** `results/human_validation.csv` ships as a
  header-only template. With no ratings the correlation is reported as skipped
  rather than filled with invented numbers.
