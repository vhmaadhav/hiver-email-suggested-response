"""Inject the measured results table into README.md from results/metrics.json.

No benchmark number in the README is ever typed by hand - this script is the
only thing allowed to write them, and it reads only generated files.

    uv run python scripts/update_readme_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

START = "<!--RESULTS_TABLE-->"
END = "<!--/RESULTS_TABLE-->"


def fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.1f}{suffix}"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def build_table(m: dict) -> str:
    cfg = m.get("config", {})
    dims = m.get("main_per_dimension_mean_1_to_5") or {}
    delta = m.get("main_minus_baseline")

    lines = [
        f"Measured on **{m['n_examples']} held-out examples** "
        f"(seed {cfg.get('seed')}, never seen by retrieval). "
        "Every value below is read directly from "
        "[`results/metrics.json`](results/metrics.json).",
        "",
        "| Metric | Main (retrieval + LLM) | Baseline (copy nearest reply) |",
        "|---|---:|---:|",
        f"| **Overall score** (0-100) | **{fmt(m['main_mean_overall_score'])}** | "
        f"{fmt(m['baseline_mean_overall_score'])} |",
        f"| Judge score (0-100) | {fmt(m['main_mean_judge_score'])} | "
        f"{fmt(m['baseline_mean_judge_score'])} |",
        f"| Semantic similarity (0-100) | {fmt(m['main_mean_semantic_similarity'])} | "
        f"{fmt(m['baseline_mean_semantic_similarity'])} |",
        f"| Acceptable rate | {pct(m['main_acceptable_rate'])} | "
        f"{pct(m['baseline_acceptable_rate'])} |",
        f"| Critical-error rate | {pct(m.get('main_critical_error_rate'))} | "
        f"{pct(m.get('baseline_critical_error_rate'))} |",
    ]
    if delta is not None:
        lines.append(f"| **Main − Baseline** (overall) | **{delta:+.1f}** | |")
    lines += [
        "",
        "Rubric dimensions for the main system (mean, 1-5):",
        "",
        "| task_fulfillment | action_alignment | completeness | tone |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join(
            fmt(dims.get(k)) for k in
            ("task_fulfillment", "action_alignment", "completeness", "tone")
        ) + " |",
        "",
        f"Generation `{cfg.get('gen_model')}` · judge `{cfg.get('judge_model')}` · "
        f"semantic `{cfg.get('semantic_model')}` ({cfg.get('semantic_backend')}) · "
        f"corpus {cfg.get('n_corpus')} emails · runtime "
        f"{cfg.get('runtime_seconds')}s on CPU.",
    ]
    if m.get("main_n_failed"):
        lines.append(
            f"\n`main_n_failed = {m['main_n_failed']}` row(s) errored and are "
            "excluded from the means rather than silently defaulted."
        )
    if cfg.get("same_model_for_gen_and_judge"):
        lines.append(
            "\n> **Judge caveat.** Generation and judging use the same model here, "
            "which invites self-preference bias. See §13. The baseline is scored by "
            "the identical judge, so the *difference* is more trustworthy than "
            "either absolute number."
        )
    hv = m.get("human_validation")
    if isinstance(hv, dict) and hv.get("status") == "ok":
        lines.append(
            f"\nHuman validation: Spearman ρ = {hv.get('spearman_rho')} over "
            f"{hv.get('n_rated')} manually rated examples; acceptable-agreement "
            f"{hv.get('acceptable_agreement')}."
        )
    else:
        lines.append(
            "\nHuman validation: no manual ratings recorded yet — "
            "`results/human_validation.csv` is an empty template, and no labels "
            "are invented to fill it (§12)."
        )
    return "\n".join(lines)


def main() -> int:
    metrics_path = ROOT / "results" / "metrics.json"
    if not metrics_path.exists():
        print(f"ERROR: {metrics_path} not found. Run scripts/run_eval.py first.",
              file=sys.stderr)
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    block = f"{START}\n{build_table(metrics)}\n{END}"

    if START in readme and END in readme:
        head, rest = readme.split(START, 1)
        _old, tail = rest.split(END, 1)
        readme = head + block + tail
    elif START in readme:
        head, tail = readme.split(START, 1)
        readme = head + block + tail
    else:
        print("ERROR: no <!--RESULTS_TABLE--> marker in README.md", file=sys.stderr)
        return 1

    readme_path.write_text(readme, encoding="utf-8")
    print(f"Injected measured results into {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
