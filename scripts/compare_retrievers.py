"""Compare TF-IDF / dense / hybrid retrieval - no LLM calls, no API cost.

Retrieval quality is measured in *reply space*: for each held-out email we
embed the gold human reply and the replies of the retrieved neighbours, and
ask how similar they are. A retriever is doing its job when the neighbours it
surfaces were answered the way this email needs to be answered - which is the
property that actually helps a grounded generator, and it is measurable
without spending a single token.

    uv run python scripts/compare_retrievers.py --eval-n 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hiver_email.data import (  # noqa: E402
    DEFAULT_SEED,
    load_pairs,
    sample_heldout,
    split_pairs,
)
from hiver_email.retrieval import build_retriever  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-n", type=int, default=200,
                   help="held-out queries to measure (default 200)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="hybrid weight on the dense score")
    p.add_argument("--kinds", default="tfidf,dense,hybrid")
    p.add_argument("--out", default=str(ROOT / "results" / "retrieval_comparison.json"))
    args = p.parse_args()

    pairs = load_pairs()
    corpus, heldout = split_pairs(pairs, seed=args.seed)
    queries = sample_heldout(heldout, args.eval_n, seed=args.seed)
    print(f"corpus {len(corpus)} | queries {len(queries)} | top_k {args.top_k}")

    corpus_ids = {p.example_id for p in corpus}
    assert not any(q.example_id in corpus_ids for q in queries), "leakage"

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                                   device="cpu")

    def embed(texts: list[str]) -> np.ndarray:
        return embedder.encode(texts, batch_size=32, convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False).astype(np.float32)

    print("Embedding gold replies ...")
    gold = embed([q.human_reply for q in queries])

    cache = str(ROOT / ".cache" / "embeddings")
    report: dict[str, dict] = {}

    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        t0 = time.time()
        retriever = build_retriever(corpus, kind=kind, alpha=args.alpha,
                                    cache_dir=cache)
        build_s = time.time() - t0

        t0 = time.time()
        top1_reply_sim, best_reply_sim, query_sim, empty = [], [], [], 0
        for i, q in enumerate(queries):
            hits = retriever.retrieve(q.retrieval_text(), top_k=args.top_k)
            if not hits:
                empty += 1
                top1_reply_sim.append(0.0)
                best_reply_sim.append(0.0)
                continue
            query_sim.append(hits[0].similarity)
            reply_vecs = embed([h.human_reply for h in hits])
            sims = reply_vecs @ gold[i]
            top1_reply_sim.append(float(sims[0]))
            best_reply_sim.append(float(sims.max()))
        search_s = time.time() - t0

        report[kind] = {
            "mean_top1_reply_similarity": round(float(np.mean(top1_reply_sim)) * 100, 2),
            "mean_best_of_k_reply_similarity": round(float(np.mean(best_reply_sim)) * 100, 2),
            "mean_query_similarity": round(float(np.mean(query_sim)) * 100, 2) if query_sim else 0.0,
            "queries_with_no_hit": empty,
            "index_build_seconds": round(build_s, 1),
            "search_seconds": round(search_s, 1),
        }
        print(f"  {kind:7s} reply-sim top1={report[kind]['mean_top1_reply_similarity']:5.2f} "
              f"best@{args.top_k}={report[kind]['mean_best_of_k_reply_similarity']:5.2f} "
              f"query-sim={report[kind]['mean_query_similarity']:5.2f} "
              f"build={build_s:.1f}s", flush=True)

    out = {
        "n_queries": len(queries),
        "top_k": args.top_k,
        "hybrid_alpha": args.alpha,
        "seed": args.seed,
        "metric": ("cosine similarity (x100) between the gold held-out human reply "
                   "and the replies attached to the retrieved neighbours, using "
                   "all-MiniLM-L6-v2; no LLM calls"),
        "results": report,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"{'Retriever':<10}{'reply-sim top1':>16}{f'best@{args.top_k}':>12}"
          f"{'query-sim':>12}{'build s':>10}")
    print("-" * 70)
    for kind, r in report.items():
        print(f"{kind:<10}{r['mean_top1_reply_similarity']:>16.2f}"
              f"{r['mean_best_of_k_reply_similarity']:>12.2f}"
              f"{r['mean_query_similarity']:>12.2f}{r['index_build_seconds']:>10.1f}")
    print("=" * 70)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
