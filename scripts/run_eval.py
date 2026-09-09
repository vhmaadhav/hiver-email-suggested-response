"""Run the full evaluation over a deterministic held-out sample.

    uv run python scripts/run_eval.py --eval-n 40

Writes results/per_response.csv and results/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Enron text contains non-cp1252 characters; do not die on a Windows console.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from hiver_email.data import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TEST_FRACTION,
    load_pairs,
    sample_heldout,
    split_pairs,
)
from hiver_email.evaluation import (  # noqa: E402
    SemanticScorer,
    human_validation,
    rows_to_frame,
    summarise,
)
from hiver_email.generator import SuggestedReplyGenerator  # noqa: E402
from hiver_email.judge import RubricJudge  # noqa: E402
from hiver_email.llm import LLMClient  # noqa: E402
from hiver_email.retrieval import baseline_reply, build_retriever  # noqa: E402
from hiver_email.schemas import (  # noqa: E402
    JUDGE_WEIGHTS,
    ResponseScore,
    overall_score,
)

RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-n", type=int, default=40,
                   help="held-out examples to evaluate (default 40)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--retriever", default="tfidf", choices=["tfidf", "dense", "hybrid"],
                   help="tfidf is the default and produced the committed results")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="hybrid only: weight on the dense score")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the retrieval-only baseline (halves judge cost)")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel LLM workers; only overlaps network waits")
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--csv", default=None, help="explicit dataset CSV path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Loading dataset ...")
    pairs = load_pairs(args.csv)
    corpus, heldout = split_pairs(pairs, seed=args.seed,
                                  test_fraction=args.test_fraction)
    eval_set = sample_heldout(heldout, args.eval_n, seed=args.seed)
    print(f"  {len(pairs)} usable pairs -> {len(corpus)} corpus / "
          f"{len(heldout)} held out; evaluating {len(eval_set)}")

    # Leakage guard: an evaluation email must never be retrievable.
    corpus_ids = {p.example_id for p in corpus}
    leaked = [p.example_id for p in eval_set if p.example_id in corpus_ids]
    if leaked:
        raise SystemExit(f"FATAL: {len(leaked)} eval examples are in the corpus: {leaked[:5]}")
    print("  leakage check: OK (no held-out example is in the retrieval corpus)")

    print(f"Building {args.retriever} index ...")
    retriever = build_retriever(
        corpus, kind=args.retriever, include_subject=True, alpha=args.alpha,
        cache_dir=str(ROOT / ".cache" / "embeddings"),
    )
    print(f"  indexed {retriever.size} historical emails")

    client = LLMClient()
    generator = SuggestedReplyGenerator(retriever, client, top_k=args.top_k)
    judge = RubricJudge(client)
    print(f"  gen model: {client.gen_model} | judge model: {client.judge_model}")

    systems = ["main"] if args.no_baseline else ["main", "baseline"]

    def evaluate_one(pair) -> list[ResponseScore]:
        """Generate + judge one held-out example. Runs on a worker thread.

        Retrieval and scoring stay deterministic; only the network waits are
        overlapped, so results do not depend on how many workers we use.
        """
        out: list[ResponseScore] = []
        result = generator.generate(pair.retrieval_text())
        retrieved = result.retrieved
        candidates = {"main": result.reply}
        if "baseline" in systems:
            candidates["baseline"] = baseline_reply(retrieved)

        for system in systems:
            reply = candidates[system]
            row = ResponseScore(
                example_id=pair.example_id,
                system=system,
                incoming_email=pair.incoming_email,
                human_reply=pair.human_reply,
                generated_reply=reply,
                retrieved_example_ids="|".join(r.example_id for r in retrieved),
                retrieval_scores="|".join(f"{r.similarity:.4f}" for r in retrieved),
                error=result.error if system == "main" else "",
            )
            if not row.error and system == "main" and not reply.strip():
                row.error = "generation_returned_empty_reply"
            if row.error:
                out.append(row)
                continue
            try:
                verdict = judge.judge(pair.incoming_email, pair.human_reply,
                                      reply, retrieved)
            except Exception as exc:  # noqa: BLE001 - recorded, never defaulted
                row.error = f"judge_failed: {type(exc).__name__}: {exc}"[:300]
                out.append(row)
                continue
            row.task_fulfillment = verdict.task_fulfillment
            row.action_alignment = verdict.action_alignment
            row.completeness = verdict.completeness
            row.tone = verdict.tone
            row.critical_error = verdict.critical_error
            row.acceptable = verdict.acceptable
            row.judge_score = round(verdict.score_100(), 2)
            row.judge_reason = verdict.reason
            out.append(row)
        return out

    results_by_id: dict[str, list[ResponseScore]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate_one, p): p for p in eval_set}
        for done, future in enumerate(as_completed(futures), start=1):
            pair = futures[future]
            try:
                batch = future.result()
            except Exception as exc:  # noqa: BLE001 - never lose an example
                batch = [ResponseScore(
                    example_id=pair.example_id, system=s,
                    incoming_email=pair.incoming_email, human_reply=pair.human_reply,
                    generated_reply="", error=f"pipeline_failed: {exc}"[:300],
                ) for s in systems]
            results_by_id[pair.example_id] = batch
            main_row = next((r for r in batch if r.system == "main"), None)
            status = "ERR" if (main_row and main_row.error) else (
                f"{main_row.judge_score:.0f}" if main_row else "-")
            print(f"  [{done}/{len(eval_set)}] {pair.example_id} judge(main)={status}",
                  flush=True)

    # Re-order deterministically: thread completion order must not leak into
    # the saved report.
    rows: list[ResponseScore] = []
    for pair in eval_set:
        rows.extend(results_by_id.get(pair.example_id, []))

    # --- Signal A, batched: one model load, one forward pass ---------------
    print("Scoring semantic reference similarity ...")
    scorer = SemanticScorer()
    print(f"  backend: {scorer.backend}"
          + (f" ({scorer.fallback_reason[:120]})" if scorer.backend != "sentence-transformers" else ""))
    live = [r for r in rows if not r.error]
    sims = scorer.score_pairs([r.generated_reply for r in live],
                              [r.human_reply for r in live])
    for row, sim in zip(live, sims):
        row.semantic_similarity = sim
        row.overall_score = round(overall_score(sim, row.judge_score), 2)

    # --- Persist -----------------------------------------------------------
    per_response = results_dir / "per_response.csv"
    rows_to_frame(rows).to_csv(per_response, index=False, encoding="utf-8")

    main_m = summarise(rows, "main")
    base_m = summarise(rows, "baseline") if "baseline" in systems else None
    hv = human_validation(rows, results_dir / "human_validation.csv", system="main")

    metrics = {
        "n_examples": main_m.n,
        "main_mean_overall_score": main_m.mean_overall_score,
        "main_mean_judge_score": main_m.mean_judge_score,
        "main_mean_semantic_similarity": main_m.mean_semantic_similarity,
        "main_acceptable_rate": main_m.acceptable_rate,
        "main_critical_error_rate": main_m.critical_error_rate,
        "main_per_dimension_mean_1_to_5": main_m.per_dimension,
        "main_n_failed": main_m.n_failed,
        "baseline_mean_overall_score": base_m.mean_overall_score if base_m else None,
        "baseline_mean_judge_score": base_m.mean_judge_score if base_m else None,
        "baseline_mean_semantic_similarity": (
            base_m.mean_semantic_similarity if base_m else None),
        "baseline_acceptable_rate": base_m.acceptable_rate if base_m else None,
        "baseline_critical_error_rate": base_m.critical_error_rate if base_m else None,
        "main_minus_baseline": (
            round(main_m.mean_overall_score - base_m.mean_overall_score, 4)
            if base_m else None),
        "human_validation": hv,
        "config": {
            "gen_model": client.gen_model,
            "judge_model": client.judge_model,
            "same_model_for_gen_and_judge": client.gen_model == client.judge_model,
            "semantic_backend": scorer.backend,
            "semantic_model": scorer.model_name,
            "seed": args.seed,
            "test_fraction": args.test_fraction,
            "top_k": args.top_k,
            "retriever": args.retriever,
            "hybrid_alpha": args.alpha if args.retriever == "hybrid" else None,
            "workers": args.workers,
            "n_pairs_total": len(pairs),
            "n_corpus": len(corpus),
            "n_heldout_pool": len(heldout),
            "judge_weights": JUDGE_WEIGHTS,
            "overall_weights": {"semantic": 0.30, "judge": 0.70},
            "runtime_seconds": round(time.time() - t0, 1),
        },
    }
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print_table(main_m, base_m, metrics)
    print(f"\nWrote {per_response}")
    print(f"Wrote {results_dir / 'metrics.json'}")
    return 0


def print_table(main_m, base_m, metrics: dict) -> None:
    def fmt(v):
        return "n/a" if v is None else f"{v:.2f}"

    w = 34
    print("\n" + "=" * 74)
    print(f"EVALUATION RESULTS  (n = {main_m.n} held-out examples)")
    print("=" * 74)
    print(f"{'Metric':<{w}}{'Main (RAG+LLM)':>19}{'Baseline (copy)':>19}")
    print("-" * 74)
    rows = [
        ("Overall score (0-100)", main_m.mean_overall_score,
         base_m.mean_overall_score if base_m else None),
        ("Judge score (0-100)", main_m.mean_judge_score,
         base_m.mean_judge_score if base_m else None),
        ("Semantic similarity (0-100)", main_m.mean_semantic_similarity,
         base_m.mean_semantic_similarity if base_m else None),
        ("Acceptable rate", main_m.acceptable_rate,
         base_m.acceptable_rate if base_m else None),
        ("Critical error rate", main_m.critical_error_rate,
         base_m.critical_error_rate if base_m else None),
    ]
    for label, a, b in rows:
        print(f"{label:<{w}}{fmt(a):>19}{fmt(b):>19}")
    print("-" * 74)
    for dim, val in main_m.per_dimension.items():
        print(f"{'  ' + dim + ' (1-5)':<{w}}{val:>19.2f}")
    print("-" * 74)
    delta = metrics.get("main_minus_baseline")
    if delta is not None:
        print(f"{'Main - Baseline (overall)':<{w}}{delta:>19.2f}")
    if main_m.n_failed or (base_m and base_m.n_failed):
        print(f"failed rows: main={main_m.n_failed} "
              f"baseline={base_m.n_failed if base_m else 0}")
    hv = metrics.get("human_validation")
    if hv is None:
        print("human validation: results/human_validation.csv not present (skipped)")
    elif hv.get("status") == "ok":
        print(f"human validation: n={hv['n_rated']} spearman={hv['spearman_rho']} "
              f"agreement={hv.get('acceptable_agreement')}")
    else:
        print(f"human validation: {hv.get('status')} (no ratings used)")
    print("=" * 74)


if __name__ == "__main__":
    raise SystemExit(main())
