"""Generate one suggested reply for an ad-hoc incoming email.

    uv run python scripts/run_demo.py --email "Could you send me the revised deck before tomorrow?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Enron text contains non-cp1252 characters; do not die on a Windows console.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from hiver_email.data import DEFAULT_SEED, load_pairs, split_pairs  # noqa: E402
from hiver_email.generator import SuggestedReplyGenerator  # noqa: E402
from hiver_email.llm import LLMClient  # noqa: E402
from hiver_email.retrieval import TfidfRetriever, baseline_reply  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--email", required=True, help="the incoming email text")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--csv", default=None)
    p.add_argument("--show-retrieved", action="store_true",
                   help="print the full retrieved examples, not just a preview")
    args = p.parse_args()

    pairs = load_pairs(args.csv)
    corpus, _heldout = split_pairs(pairs, seed=args.seed)
    retriever = TfidfRetriever(corpus, include_subject=True)
    print(f"Indexed {retriever.size} historical emails (80% split, held-out excluded).\n")

    client = LLMClient()
    generator = SuggestedReplyGenerator(retriever, client, top_k=args.top_k)
    result = generator.generate(args.email)

    print("=" * 72)
    print("INCOMING EMAIL")
    print("=" * 72)
    print(args.email)

    print("\n" + "=" * 72)
    print(f"RETRIEVED GROUNDING (top {args.top_k}, TF-IDF cosine)")
    print("=" * 72)
    if not result.retrieved:
        print("(nothing similar found)")
    limit = 100000 if args.show_retrieved else 240
    for i, ex in enumerate(result.retrieved, start=1):
        print(f"\n[{i}] id={ex.example_id}  similarity={ex.similarity:.4f}")
        print(f"    incoming: {_one_line(ex.incoming_email, limit)}")
        print(f"    reply   : {_one_line(ex.human_reply, limit)}")

    print("\n" + "=" * 72)
    print("BASELINE (nearest historical reply, copied verbatim)")
    print("=" * 72)
    print(_one_line(baseline_reply(result.retrieved), limit) or "(none)")

    print("\n" + "=" * 72)
    print(f"SUGGESTED REPLY (generated, model={result.model})")
    print("=" * 72)
    if result.error:
        print(f"GENERATION FAILED: {result.error}")
        return 1
    print(result.reply)
    return 0


def _one_line(text: str, limit: int) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= limit else t[:limit] + " ..."


if __name__ == "__main__":
    raise SystemExit(main())
