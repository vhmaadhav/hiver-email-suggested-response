# Hiver Email Suggested-Response System

Given a new incoming email, retrieve the most similar historical email/reply pairs and use them to ground an LLM that drafts a suggested reply — then score every draft with a two-signal evaluator that measures whether the reply is *right*, not whether it matches the reference word-for-word.

## Results

<!--RESULTS_TABLE-->

## Run it

```bash
uv sync
```

```bash
uv run python scripts/fetch_data.py
```

```bash
cp .env.example .env   # then add your key
```

```bash
uv run python scripts/run_demo.py --email "Could you send me the revised deck before tomorrow?"
```

```bash
uv run python scripts/run_eval.py --eval-n 40
```

```bash
uv run pytest
```

`run_eval.py` writes [`results/per_response.csv`](results/per_response.csv) and [`results/metrics.json`](results/metrics.json) and prints the table above. Useful flags: `--workers N` (parallel LLM calls, default 6), `--eval-n N`, `--seed N`, `--no-baseline`.

---

## 1. Problem framing

A support or business inbox answers the same *kinds* of questions repeatedly. The useful signal is not a general-purpose writing model — it is the organisation's own history of how these emails were actually answered. So the system is retrieval-grounded generation:

```
new incoming email
   → TF-IDF over historical incoming emails
   → top-3 (historical email, human reply, similarity)
   → LLM prompted with those as guidance
   → suggested reply
```

The hard part is not producing text. It is knowing whether the text is any good. Most of the engineering here went into the evaluator.

## 2. Dataset and why it was chosen

[`oanannv/enron-email-reply-dataset`](https://www.kaggle.com/datasets/oanannv/enron-email-reply-dataset) — ~15,377 cleaned email/reply pairs derived from the public **Enron Email Dataset** (released by FERC, distributed for research by [CMU](https://www.cs.cmu.edu/~enron/)).

It was chosen because it is the rare public corpus that has the thing this task actually needs: **real incoming emails paired with the real replies humans sent to them.** Most email corpora are single messages with no reply, which gives you nothing to ground on and nothing to evaluate against.

Loading applies light cleaning (see `src/hiver_email/data.py`): normalise newlines, drop pairs where the email is under 40 characters or the reply under 20 (auto-acknowledgements and fragments), truncate at 4,000 characters, and collapse exact duplicate pairs by a content hash. That takes 15,377 raw rows to **11,914 usable pairs**.

Column names are resolved against a candidate list rather than hardcoded, so a slightly different dump of the dataset still loads.

## 3. Data limitations

These matter and should not be glossed over.

- **Enron is business email, not customer support.** Internal colleagues with shared context, not a support agent answering a stranger. Absolute scores here will not transfer to a Hiver-style support inbox; the *evaluation method* is what transfers.
- **Heavy domain skew.** Energy trading, gas nominations, legal redlines. Retrieval quality on a support corpus would differ.
- **~2001 vintage.** Conventions, tooling and tone have moved on.
- **A reply is not proof of resolution.** The dataset records what was sent, not whether it solved anything. A reference reply is evidence of *one plausible answer*, which is precisely why the evaluator treats it as one valid answer rather than as ground truth.
- **Missing thread context.** Many replies depend on prior messages, attachments or hallway conversations the model cannot see. Some references are unreproducible from the incoming email alone — this puts a real ceiling on achievable scores and is a reason to read the per-dimension breakdown, not just the headline.
- **Real people's private mail,** made public through litigation. Used here strictly as a research corpus.

## 4. Architecture

```
src/hiver_email/
  schemas.py     pydantic contracts + scoring arithmetic (single source of truth for weights)
  data.py        load, normalise, deterministic 80/20 split
  retrieval.py   TF-IDF index + top-k + the retrieval-only baseline
  llm.py         one OpenAI-SDK client, bounded retries, strict JSON parsing
  generator.py   grounded prompt -> suggested reply
  judge.py       rubric prompt -> structured verdict
  evaluation.py  semantic scorer, aggregation, human-validation correlation
```

Deliberately boring. No FAISS, no vector database, no LangChain, no LlamaIndex, no web framework. At ~9.5 k documents, retrieval is one sparse matrix and a dot product; anything heavier would add dependencies and hide the mechanism without making results better.

**Provider handling.** One integration, the OpenAI SDK, with `OPENAI_BASE_URL` configurable — so any OpenAI-compatible endpoint works by changing an environment variable. During this build the provider changed once and the model changed twice; every switch was a `.env` edit, not a code change. `GEN_MODEL` and `JUDGE_MODEL` are independent so gen and judge can be different models.

**16 GB CPU-only.** `pyproject.toml` pins the CPU-only torch wheel so `sentence-transformers` cannot drag in a multi-GB CUDA runtime. The TF-IDF index is `float32`. The embedding model (all-MiniLM-L6-v2, ~90 MB) is loaded once and scores the whole run in one batched pass. Peak observed memory for a full evaluation: **~232 MB**.

## 5. Retrieval grounding

`TfidfVectorizer` over `subject + body` of historical **incoming** emails: lowercase, English stop words, unigrams + bigrams, `sublinear_tf`, `min_df=2`, capped at 50 k features. Rows are L2-normalised, so a dot product *is* cosine similarity. Top-3 neighbours are returned, each exposing the historical incoming email, the human reply, and the similarity score. Zero-similarity hits are dropped rather than padded.

The retrieved pairs go into the prompt as guidance. The prompt instructs the model to answer the actual new email rather than copy an example, to preserve tone, to avoid fabricating commitments, dates, prices, approvals, attachments or completed actions, to acknowledge uncertainty instead of inventing detail, and to return only the reply body.

**Retrieval quality is genuinely limited** — typical top-1 cosine on this corpus is ~0.2. Enron's vocabulary is broad and TF-IDF is lexical. This is a known ceiling, visible in the per-response similarity scores, and the honest first upgrade would be dense retrieval.

## 6. Baseline

`new email → nearest TF-IDF historical email → return that human reply verbatim.`

Both systems are scored by the identical evaluator on the identical examples. This is the comparison that makes the headline number mean something: it answers **"does LLM synthesis actually beat copying the closest previous answer?"** — the question a reviewer should ask of any RAG system. Without it, a good-looking absolute score is unfalsifiable.

## 7. What "accuracy" means for an email response

There is no single correct reply to an email. Two replies with no words in common can both be excellent. So "accuracy" cannot mean string agreement. What it should mean:

1. **Does it do the job the sender asked for?**
2. **Does it commit to the right things — and only those?** Inventing an approval, a price or a completed action is worse than being vague, because someone may send it.
3. **Is it complete enough to be useful?**
4. **Is it appropriate in tone?**

The rubric is these four questions, in that order of weight, plus a hard flag for the failure that makes a draft dangerous rather than merely weak.

## 8. Evaluation metric

Every response gets **two independent signals** that fail in different directions.

**Signal A — semantic reference similarity.** Cosine between the generated reply and the held-out human reply, embedded with local `all-MiniLM-L6-v2`, clamped to [0,1] and mapped to 0–100. Deterministic, offline, no LLM in the loop. Negatives are clamped rather than rescaled — rescaling would award ~50/100 to unrelated text. An empty reply scores 0. A clearly-labelled TF-IDF cosine fallback exists so a dependency problem degrades the metric visibly instead of blocking the run; `metrics.json` records which backend produced the numbers.

**Signal B — LLM rubric judge.** The judge sees the incoming email, the reference human reply, the candidate, and the retrieved grounding. It is told explicitly: *"the historical human response is ONE valid response, not wording that must be copied."* Without that instruction the judge collapses into a paraphrase detector and the rubric is pointless. It returns strict JSON — four 1–5 scores plus `critical_error`, `acceptable`, and a `reason`.

| Dimension | Weight | Question |
|---|---|---|
| `task_fulfillment` | 35% | Does it address what the email requires? |
| `action_alignment` | 30% | Does it preserve the reference's commitments without inventing new ones? |
| `completeness` | 20% | Does it cover what an effective reply needs? |
| `tone` | 15% | Clear, concise, professional, appropriate? |

Each 1–5 score maps linearly to 0–100 as `(v-1)/4*100`, so the floor of the rubric is the floor of the score. **If `critical_error` is true, the judge score is capped at 40** — contradicting the email, inventing consequential facts, falsely claiming an attachment or completed action, or missing the requested action entirely. A fluent, well-toned reply that fabricates a commitment must not be able to score well, and the cap enforces that no matter how the other dimensions land.

```
overall_score = 0.30 * semantic_similarity + 0.70 * judge_score
```

**This weighting is not claimed to be optimal.** It is a defensible prior, not a tuned parameter, and nothing was fitted to produce it. The judge gets more weight because semantic similarity unfairly penalises valid paraphrases — the most common way a genuinely good reply looks bad. Semantic similarity is kept at 30% precisely because it is deterministic, reference-anchored and cannot be talked into a good score by fluent prose: it is the check on the judge. Both components are stored per response, so anyone can re-weight from `per_response.csv` without re-running a single LLM call.

## 9. Why exact match / BLEU-style overlap is insufficient

> Incoming: *"Can you send the revised deck before tomorrow?"*
> Reference: *"Sure — sending it tonight."*
> Candidate: *"Yes, I'll have the updated version over to you this evening."*

BLEU/ROUGE score this near zero on content words. It is a perfect reply. Overlap metrics measure *wording*, and wording is the one thing that legitimately varies most between two equally good emails.

Worse, the failure is asymmetric in the dangerous direction: a reply that copies the reference's phrasing but inverts a commitment ("I *won't* have it tonight") scores *high* on overlap. Overlap metrics are blind to exactly the errors that matter most — invented commitments and contradictions — while punishing exactly the variation that is fine. Hence a rubric that scores substance, with a hard cap for the errors that make a draft unsendable.

## 10. Per-response evaluation

Every row in [`results/per_response.csv`](results/per_response.csv) carries: `example_id`, `system` (`main` / `baseline`), `incoming_email`, `human_reply`, `generated_reply`, `retrieved_example_ids`, `retrieval_scores`, `semantic_similarity`, the four rubric dimensions, `critical_error`, `acceptable`, `judge_score`, `overall_score`, `judge_reason`, and `error`.

Every aggregate is recomputable from this file. The `judge_reason` column means every score can be spot-checked against a stated rationale rather than taken on trust.

**Failures are recorded, never defaulted.** If the model returns nothing, or the judge emits malformed JSON, the row is saved with an `error` and excluded from the means, and the count surfaces as `main_n_failed`. Silently substituting a default verdict would fabricate a measurement. `parse_json_object()` tolerates markdown fences and leading prose but raises on anything that is not a JSON object — tested in `tests/test_scoring.py`.

## 11. Overall evaluation

[`results/metrics.json`](results/metrics.json) holds `n_examples`, the main system's mean overall / judge / semantic scores, acceptable rate, critical-error rate, per-dimension means, the same for the baseline, `main_minus_baseline`, and a `config` block recording models, seed, split sizes, weights, semantic backend and runtime — so any number in this README is traceable to a generated file and a reproducible configuration.

Default evaluation is a **deterministic sample of 40 held-out examples**, seeded, so a run is fast and cheap. The full held-out pool is available via `--eval-n`.

**No held-out example ever enters retrieval.** The split is a seeded permutation over content-addressed ids sorted for stability, so it does not depend on CSV row order. `run_eval.py` asserts the eval set and corpus are disjoint before any LLM call and aborts if not, and `tests/test_retrieval.py` verifies held-out emails are unreachable from the index by id *and* by text.

## 12. Human validation

Drop ratings into [`results/human_validation.csv`](results/human_validation.csv) (ships as a header-only template):

```csv
example_id,human_score,human_acceptable
```

`run_eval.py` then reports the **Spearman correlation** between your 0–100 scores and the automatic overall score, plus **binary agreement** between your `human_acceptable` and the judge's.

This is the check that the evaluator is worth trusting. An automatic metric is only credible if it tracks human judgement, and **rating 10–15 generated examples by hand is a cheap sanity check on that** — it will not prove the metric is calibrated, but it will quickly expose a metric that is anti-correlated with what a person considers a good reply.

**No labels are fabricated.** With the template empty, the correlation is reported as skipped (`status: empty`) rather than filled with invented numbers. The ratings in this repo's committed results are whatever is actually in that file — currently none.

## 13. Limitations of LLM-as-a-judge

**LLM judges are imperfect and carry known biases** — verbosity bias (longer answers read as better), position and self-preference bias, sensitivity to prompt phrasing, and imperfect run-to-run consistency. Judge temperature is 0 here, which reduces but does not eliminate variance.

**Generation and judging currently use the same model** (`openai/gpt-oss-20b`). This is a genuine limitation and it is not a design preference: of twelve models probed on this endpoint, only two were reachable, and the other could neither return clean prose nor emit valid JSON (see [FAILURES.md](FAILURES.md) §6–9). A model evaluating its own output invites self-preference bias, which likely inflates the main system's scores relative to the baseline's. **In a production evaluation, use a different model — ideally a different vendor — for judging.** The code already supports this: set `JUDGE_MODEL` to something else. `metrics.json` records `config.same_model_for_gen_and_judge` so no reader has to take this on trust.

Two further guards: the baseline is judged by the identical judge on the identical examples, so shared bias affects both sides of the comparison and the *difference* is more trustworthy than either absolute; and the deterministic semantic signal keeps 30% of the score outside the judge's reach entirely.

The judge is a screening instrument, not a source of truth. That is what §12 is for.

## 14. Tradeoffs

| Decision | Why | What it costs |
|---|---|---|
| TF-IDF, not embeddings, for retrieval | Explainable, instant, no index to manage, no GPU | Lexical only; ~0.2 top-1 cosine on this corpus. Dense retrieval is the first upgrade |
| 40-example default eval | Fast and affordable in a timed run | Wide confidence intervals — a few points of difference is not a real difference at n=40 |
| Judge weighted 0.70 | Semantic similarity punishes valid paraphrase | Inherits judge bias; mitigated by the deterministic 30% and the shared-judge baseline |
| One provider integration | `OPENAI_BASE_URL` covers compatible endpoints; provider changed twice during this build with zero code change | Non-compatible providers need an adapter |
| Fail loudly on bad judge JSON | A defaulted score is a fabricated measurement | Some rows drop out; reported as `main_n_failed` |
| Enron | Only realistic public corpus with real reply pairs | Not support email; absolute numbers do not transfer |
| Threaded eval loop | 40 examples × 3 LLM calls is ~25 min serial | Concurrency only overlaps network waits; ordering is restored before saving |

## 15. AI tools used

This project was built in a timed challenge with **Claude Code (Claude Opus 5)** as an active pair-programmer. Concretely:

- **Scaffolding and implementation.** Module layout, prompts, scoring arithmetic, CLI scripts and tests were drafted with Claude and then reviewed, corrected and run by hand.
- **Debugging.** Every failure in [FAILURES.md](FAILURES.md) was diagnosed interactively — the empty-completion bug from reasoning tokens, the leaked chain-of-thought, the judge's JSON failure, the Windows encoding crash.
- **What was not delegated.** The evaluation design — two independent signals, the rubric dimensions and their weights, the critical-error cap, the leakage guard, the decision to fail loudly rather than default a score — are the deliberate choices this submission stands on, and each is defended above and tested in `tests/`.
- **Verification.** No number in this README was typed by hand or by a model. All values come from `results/metrics.json`, generated by `scripts/run_eval.py` and injected into the table above. The test suite is the check on the arithmetic.

At runtime the system uses `openai/gpt-oss-20b` via the NVIDIA build API's OpenAI-compatible endpoint for generation and judging, and local `sentence-transformers/all-MiniLM-L6-v2` for the semantic signal.

## 16. Repository structure

```
.
├── README.md
├── FAILURES.md            what broke, why, and what changed in response
├── pyproject.toml         deps + CPU-only torch pin
├── .env.example
├── .gitignore             .env and data/ are never committed
├── data/
│   └── README.md          dataset source, licence, schema
├── src/hiver_email/
│   ├── schemas.py         contracts + scoring arithmetic
│   ├── data.py            load, clean, deterministic split
│   ├── retrieval.py       TF-IDF index + baseline
│   ├── llm.py             provider client + strict JSON parsing
│   ├── generator.py       grounded generation
│   ├── judge.py           rubric judge
│   └── evaluation.py      semantic signal, aggregation, human validation
├── scripts/
│   ├── fetch_data.py
│   ├── run_demo.py
│   └── run_eval.py
├── tests/
│   ├── test_data.py       split determinism, order-independence, no leakage
│   ├── test_retrieval.py  nearest-neighbour, baseline, held-out unreachable
│   └── test_scoring.py    weights, critical-error cap, malformed JSON, mocked LLMs
└── results/
    ├── per_response.csv
    ├── metrics.json
    └── human_validation.csv
```

## Licence and attribution

Code in this repository is provided for review as a hiring-challenge submission. The dataset is not redistributed here — `scripts/fetch_data.py` downloads it from Kaggle. See [`data/README.md`](data/README.md) for source, provenance and licensing; check the Kaggle dataset page for its terms before redistributing.
