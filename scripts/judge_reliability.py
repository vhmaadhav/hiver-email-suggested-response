"""Measure the judge's test-retest reliability on an existing run.

"LLM judges are imperfect" is a disclaimer. This turns it into a number.

We re-judge responses that were already scored in results/per_response.csv,
with identical inputs, and compare the two independent verdicts. Temperature
is 0, so any disagreement is irreducible sampling noise in the judge itself -
which sets a floor on how small a difference the whole harness can resolve.

    uv run python scripts/judge_reliability.py --n 30
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from hiver_email.judge import RubricJudge  # noqa: E402
from hiver_email.llm import LLMClient  # noqa: E402

DIMENSIONS = ("task_fulfillment", "action_alignment", "completeness", "tone")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(ROOT / "results"))
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--out", default=str(ROOT / "results" / "judge_reliability.json"))
    args = p.parse_args()

    csv_path = Path(args.results_dir) / "per_response.csv"
    with open(csv_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["system"] == "main" and not r["error"]][: args.n]
    if not rows:
        raise SystemExit(f"no scored main rows in {csv_path}")
    print(f"Re-judging {len(rows)} previously scored responses ...")

    client = LLMClient()
    judge = RubricJudge(client)
    print(f"  judge model: {client.judge_model} (temperature 0)")

    def rejudge(row: dict) -> tuple[dict, object | None, str]:
        try:
            return row, judge.judge(row["incoming_email"], row["human_reply"],
                                    row["generated_reply"], []), ""
        except Exception as exc:  # noqa: BLE001
            return row, None, f"{type(exc).__name__}: {exc}"[:200]

    first_scores, second_scores, failures = [], [], 0
    accept_agree, critical_agree = [], []
    dim_deltas: dict[str, list[float]] = {d: [] for d in DIMENSIONS}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(rejudge, r) for r in rows]
        for done, fut in enumerate(as_completed(futures), start=1):
            row, verdict, err = fut.result()
            if verdict is None:
                failures += 1
                print(f"  [{done}/{len(rows)}] {row['example_id']} FAILED {err}",
                      flush=True)
                continue
            a, b = float(row["judge_score"]), verdict.score_100()
            first_scores.append(a)
            second_scores.append(b)
            accept_agree.append(
                1.0 if (row["acceptable"].strip().lower() == "true") == verdict.acceptable
                else 0.0
            )
            critical_agree.append(
                1.0 if (row["critical_error"].strip().lower() == "true") == verdict.critical_error
                else 0.0
            )
            for d in DIMENSIONS:
                if row[d]:
                    dim_deltas[d].append(abs(int(float(row[d])) - getattr(verdict, d)))
            print(f"  [{done}/{len(rows)}] {row['example_id']} {a:5.1f} -> {b:5.1f} "
                  f"(Δ{b - a:+.1f})", flush=True)

    if len(first_scores) < 3:
        raise SystemExit("too few successful re-judgements to summarise")

    deltas = [b - a for a, b in zip(first_scores, second_scores)]
    abs_deltas = [abs(d) for d in deltas]
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson = round(float(pearsonr(first_scores, second_scores)[0]), 4)
        spearman = round(float(spearmanr(first_scores, second_scores)[0]), 4)
    except Exception:  # noqa: BLE001
        pearson = spearman = None

    exact = sum(1 for d in abs_deltas if d < 0.01) / len(abs_deltas)
    within10 = sum(1 for d in abs_deltas if d <= 10) / len(abs_deltas)

    out = {
        "n_rejudged": len(first_scores),
        "n_failed": failures,
        "judge_model": client.judge_model,
        "temperature": 0.0,
        "mean_judge_score_run1": round(st.mean(first_scores), 2),
        "mean_judge_score_run2": round(st.mean(second_scores), 2),
        "mean_signed_difference": round(st.mean(deltas), 2),
        "mean_absolute_difference": round(st.mean(abs_deltas), 2),
        "median_absolute_difference": round(st.median(abs_deltas), 2),
        "max_absolute_difference": round(max(abs_deltas), 2),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "identical_score_rate": round(exact, 4),
        "within_10_points_rate": round(within10, 4),
        "acceptable_flag_agreement": round(st.mean(accept_agree), 4),
        "critical_error_flag_agreement": round(st.mean(critical_agree), 4),
        "mean_abs_dimension_delta_1_to_5": {
            d: round(st.mean(v), 3) for d, v in dim_deltas.items() if v
        },
        "interpretation": (
            "Mean absolute difference is the judge's own noise floor on a single "
            "response at temperature 0. A run-level difference much smaller than "
            "this divided by sqrt(n) should not be treated as a real effect."
        ),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"JUDGE TEST-RETEST RELIABILITY  (n = {out['n_rejudged']}, temperature 0)")
    print("=" * 66)
    print(f"  mean score run 1 / run 2      {out['mean_judge_score_run1']:.2f} / "
          f"{out['mean_judge_score_run2']:.2f}")
    print(f"  mean absolute difference      {out['mean_absolute_difference']:.2f} points")
    print(f"  median / max abs difference   {out['median_absolute_difference']:.1f} / "
          f"{out['max_absolute_difference']:.1f}")
    print(f"  correlation (pearson r)       {out['pearson_r']}")
    print(f"  identical score               {out['identical_score_rate'] * 100:.0f}%")
    print(f"  within 10 points              {out['within_10_points_rate'] * 100:.0f}%")
    print(f"  acceptable flag agreement     {out['acceptable_flag_agreement'] * 100:.0f}%")
    print(f"  critical_error flag agreement {out['critical_error_flag_agreement'] * 100:.0f}%")
    print("=" * 66)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
