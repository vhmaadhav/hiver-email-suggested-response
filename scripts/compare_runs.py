"""Paired comparison of two evaluation runs.

Comparing the means of two runs is the wrong test when the runs scored
different example sets: the difference then mixes a real effect with which
examples happened to succeed. This pairs by `example_id` and reports the
per-example difference, its confidence interval, and a win/loss count -
which is what decides whether a change is worth shipping.

    uv run python scripts/compare_runs.py results/ results/hybrid/ --labels tfidf hybrid
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")


def load_main_rows(results_dir: Path) -> dict[str, dict]:
    path = Path(results_dir) / "per_response.csv"
    if not path.exists():
        raise SystemExit(f"no per_response.csv in {results_dir}")
    with open(path, encoding="utf-8") as fh:
        return {
            r["example_id"]: r
            for r in csv.DictReader(fh)
            if r["system"] == "main" and not r["error"]
        }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("baseline_dir")
    p.add_argument("candidate_dir")
    p.add_argument("--labels", nargs=2, default=["baseline", "candidate"])
    p.add_argument("--metric", default="overall_score")
    p.add_argument("--out", default=str(ROOT / "results" / "retriever_ab.json"))
    args = p.parse_args()

    a = load_main_rows(Path(args.baseline_dir))
    b = load_main_rows(Path(args.candidate_dir))
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        raise SystemExit(f"only {len(common)} paired examples - nothing to compare")

    xa = [float(a[k][args.metric]) for k in common]
    xb = [float(b[k][args.metric]) for k in common]
    diffs = [y - x for x, y in zip(xa, xb)]
    mean_d = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(len(diffs))
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)

    out = {
        "metric": args.metric,
        "labels": args.labels,
        "n_paired": len(common),
        "n_scored": {args.labels[0]: len(a), args.labels[1]: len(b)},
        f"mean_{args.labels[0]}": round(st.mean(xa), 2),
        f"mean_{args.labels[1]}": round(st.mean(xb), 2),
        "paired_mean_difference": round(mean_d, 2),
        "paired_sd_of_difference": round(st.stdev(diffs), 2),
        "standard_error": round(se, 2),
        "t_statistic": round(mean_d / se, 2) if se else None,
        "ci95": [round(mean_d - 1.96 * se, 2), round(mean_d + 1.96 * se, 2)],
        "per_example": {"better": wins, "worse": losses,
                        "tie": len(diffs) - wins - losses},
        "identical_retrieval": sum(
            1 for k in common
            if a[k]["retrieved_example_ids"] == b[k]["retrieved_example_ids"]
        ),
        "conclusion": (
            "difference is not distinguishable from noise at this sample size"
            if abs(mean_d) < 1.96 * se
            else "difference is statistically distinguishable at this sample size"
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    lo, hi = out["ci95"]
    print(f"\nPaired on {len(common)} examples ({args.metric})")
    print(f"  {args.labels[0]:<10} {st.mean(xa):.2f}")
    print(f"  {args.labels[1]:<10} {st.mean(xb):.2f}")
    print(f"  difference {mean_d:+.2f}   95% CI [{lo:+.2f}, {hi:+.2f}]   "
          f"t = {out['t_statistic']}")
    print(f"  per-example: {wins} better / {losses} worse / "
          f"{out['per_example']['tie']} tie")
    print(f"  retrieval identical on {out['identical_retrieval']}/{len(common)}")
    print(f"  -> {out['conclusion']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
