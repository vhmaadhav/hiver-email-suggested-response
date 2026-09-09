"""The evaluation harness - the part that has to be trustworthy.

Two independent signals per response:
  A. semantic reference similarity (local, deterministic, no LLM)
  B. an LLM rubric judge (substance, not wording)

They fail in different directions, which is the point: A cannot be talked
into a good score by fluent prose, and B does not punish a valid paraphrase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .schemas import ResponseScore, overall_score

DEFAULT_ST_MODEL = os.environ.get(
    "SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------------------------------
# Signal A: semantic reference similarity
# --------------------------------------------------------------------------
class SemanticScorer:
    """Cosine similarity between generated and reference reply, mapped to 0-100.

    Uses a small local sentence-transformer (all-MiniLM-L6-v2, ~90 MB, runs on
    CPU in a few seconds for 80 texts). If the model cannot be loaded we fall
    back to a clearly-labelled TF-IDF cosine so the pipeline never blocks -
    self.backend records which one actually produced the numbers.
    """

    def __init__(self, model_name: str = DEFAULT_ST_MODEL, allow_fallback: bool = True):
        self.model_name = model_name
        self.backend = "sentence-transformers"
        self.fallback_reason = ""
        self._model = None
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise
            self.backend = "tfidf-fallback"
            self.fallback_reason = str(exc)

    def score_pairs(self, generated: list[str], references: list[str]) -> list[float]:
        """Return one 0-100 score per (generated, reference) pair."""
        if len(generated) != len(references):
            raise ValueError("generated/reference length mismatch")
        if not generated:
            return []
        if self._model is not None:
            return self._score_st(generated, references)
        return self._score_tfidf(generated, references)

    def _score_st(self, generated: list[str], references: list[str]) -> list[float]:
        texts = [t if t.strip() else " " for t in generated + references]
        # batch_size kept small: this is a CPU-only, 16 GB machine.
        emb = self._model.encode(
            texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        n = len(generated)
        sims = np.sum(emb[:n] * emb[n:], axis=1)
        return [
            _to_100(float(s), blank=not generated[i].strip())
            for i, s in enumerate(sims)
        ]

    def _score_tfidf(self, generated: list[str], references: list[str]) -> list[float]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, dtype=np.float32)
        matrix = vec.fit_transform(
            [t if t.strip() else " " for t in generated + references]
        )
        n = len(generated)
        sims = np.asarray(matrix[:n].multiply(matrix[n:]).sum(axis=1)).ravel()
        return [
            _to_100(float(s), blank=not generated[i].strip())
            for i, s in enumerate(sims)
        ]


def _to_100(cosine: float, blank: bool = False) -> float:
    """Map cosine similarity onto a readable 0-100 scale.

    Cosine over normalised sentence embeddings is in [-1, 1] in principle but
    effectively [0, 1] for English prose, so we clamp negatives to 0 rather
    than rescaling - rescaling would hand ~50/100 to unrelated text.
    """
    if blank:
        return 0.0
    return round(max(0.0, min(1.0, cosine)) * 100.0, 2)


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------
@dataclass
class SystemMetrics:
    system: str
    n: int = 0
    mean_overall_score: float = 0.0
    mean_judge_score: float = 0.0
    mean_semantic_similarity: float = 0.0
    acceptable_rate: float = 0.0
    critical_error_rate: float = 0.0
    n_failed: int = 0
    per_dimension: dict[str, float] = field(default_factory=dict)


def summarise(rows: list[ResponseScore], system: str) -> SystemMetrics:
    """Aggregate scored rows for one system.

    Rows that errored are excluded from the means but reported as n_failed,
    so a failure can never quietly improve the headline number.
    """
    scored = [r for r in rows if r.system == system and not r.error]
    failed = [r for r in rows if r.system == system and r.error]
    m = SystemMetrics(system=system, n=len(scored), n_failed=len(failed))
    if not scored:
        return m
    m.mean_overall_score = _mean([r.overall_score for r in scored])
    m.mean_judge_score = _mean([r.judge_score for r in scored])
    m.mean_semantic_similarity = _mean([r.semantic_similarity for r in scored])
    m.acceptable_rate = _mean([1.0 if r.acceptable else 0.0 for r in scored])
    m.critical_error_rate = _mean([1.0 if r.critical_error else 0.0 for r in scored])
    for dim in ("task_fulfillment", "action_alignment", "completeness", "tone"):
        vals = [getattr(r, dim) for r in scored if getattr(r, dim) is not None]
        if vals:
            m.per_dimension[dim] = _mean([float(v) for v in vals])
    return m


def _mean(values: list[float]) -> float:
    return round(float(np.mean(values)), 4) if values else 0.0


def rows_to_frame(rows: list[ResponseScore]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in rows])


# --------------------------------------------------------------------------
# Human validation
# --------------------------------------------------------------------------
HUMAN_VALIDATION_COLUMNS = ["example_id", "human_score", "human_acceptable"]


def human_validation(
    rows: list[ResponseScore], path: str | Path, system: str = "main"
) -> dict | None:
    """Correlate manual ratings against the automatic score, if any exist.

    Returns None when the file is missing, and a status dict when it exists
    but holds nothing usable. We never invent labels - an empty file simply
    means this check did not run.
    """
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    missing = [c for c in HUMAN_VALIDATION_COLUMNS if c not in df.columns]
    if missing:
        return {"status": "invalid", "missing_columns": missing}
    df = df.dropna(subset=["example_id", "human_score"])
    if df.empty:
        return {"status": "empty", "n_rated": 0}

    auto = {r.example_id: r for r in rows if r.system == system and not r.error}
    human_scores: list[float] = []
    auto_scores: list[float] = []
    agree: list[float] = []
    for row in df.itertuples(index=False):
        rec = auto.get(str(row.example_id))
        if rec is None:
            continue
        human_scores.append(float(row.human_score))
        auto_scores.append(float(rec.overall_score))
        ha = getattr(row, "human_acceptable", None)
        if ha is not None and not pd.isna(ha) and str(ha).strip() != "":
            agree.append(1.0 if _as_bool(ha) == bool(rec.acceptable) else 0.0)

    if len(human_scores) < 3:
        return {
            "status": "insufficient",
            "n_rated": len(human_scores),
            "note": "need at least 3 matched ratings before a correlation is meaningful",
        }
    rho, pval = spearmanr(human_scores, auto_scores)
    out: dict = {
        "status": "ok",
        "n_rated": len(human_scores),
        "spearman_rho": None if np.isnan(rho) else round(float(rho), 4),
        "spearman_p_value": None if np.isnan(pval) else round(float(pval), 4),
    }
    if agree:
        out["acceptable_agreement"] = round(float(np.mean(agree)), 4)
        out["n_acceptable_compared"] = len(agree)
    return out


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


__all__ = [
    "SemanticScorer",
    "SystemMetrics",
    "summarise",
    "rows_to_frame",
    "human_validation",
    "overall_score",
    "HUMAN_VALIDATION_COLUMNS",
]
