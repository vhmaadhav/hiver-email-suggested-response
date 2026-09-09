"""Typed contracts shared across the pipeline.

Everything that crosses a module boundary (or gets written to disk) is a
pydantic model so that a malformed LLM payload fails loudly and early.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

# --- rubric weights -------------------------------------------------------
# Documented in the README; kept here so tests can assert against one source.
JUDGE_WEIGHTS: dict[str, float] = {
    "task_fulfillment": 0.35,
    "action_alignment": 0.30,
    "completeness": 0.20,
    "tone": 0.15,
}
CRITICAL_ERROR_SCORE_CAP = 40.0

SEMANTIC_WEIGHT = 0.30
JUDGE_WEIGHT = 0.70


class EmailPair(BaseModel):
    """One historical incoming email paired with the human reply it received."""

    example_id: str
    subject_send: str = ""
    incoming_email: str
    subject_reply: str = ""
    human_reply: str
    sender: str = ""
    recipient: str = ""
    date_send: str = ""

    def retrieval_text(self) -> str:
        """Text that gets indexed / queried by TF-IDF."""
        return f"{self.subject_send}\n{self.incoming_email}".strip()


class RetrievedExample(BaseModel):
    """A neighbour returned by the TF-IDF retriever."""

    example_id: str
    incoming_email: str
    human_reply: str
    similarity: float


class GenerationResult(BaseModel):
    reply: str
    retrieved: list[RetrievedExample] = Field(default_factory=list)
    model: str = ""
    error: str = ""


class JudgeVerdict(BaseModel):
    """Structured output of the LLM rubric judge (1-5 per dimension)."""

    task_fulfillment: int
    action_alignment: int
    completeness: int
    tone: int
    critical_error: bool
    acceptable: bool
    reason: str = ""

    @field_validator("task_fulfillment", "action_alignment", "completeness", "tone")
    @classmethod
    def _in_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError(f"rubric score {v} outside 1-5")
        return v

    def score_100(self) -> float:
        """Weighted rubric score out of 100, capped when a critical error is flagged.

        Each 1-5 dimension is mapped linearly onto 0-100 ((v - 1) / 4 * 100)
        so that the floor of the rubric scale is the floor of the score.
        """
        raw = sum(
            JUDGE_WEIGHTS[dim] * ((getattr(self, dim) - 1) / 4.0) * 100.0
            for dim in JUDGE_WEIGHTS
        )
        if self.critical_error:
            return min(raw, CRITICAL_ERROR_SCORE_CAP)
        return raw


class ResponseScore(BaseModel):
    """Everything we record for a single evaluated response."""

    example_id: str
    system: str  # "main" or "baseline"
    incoming_email: str
    human_reply: str
    generated_reply: str
    retrieved_example_ids: str = ""
    retrieval_scores: str = ""
    semantic_similarity: float = 0.0
    task_fulfillment: int | None = None
    action_alignment: int | None = None
    completeness: int | None = None
    tone: int | None = None
    critical_error: bool | None = None
    acceptable: bool | None = None
    judge_score: float = 0.0
    overall_score: float = 0.0
    judge_reason: str = ""
    error: str = ""


def overall_score(semantic_similarity: float, judge_score: float) -> float:
    """Blend the two independent signals.

    Not claimed to be optimal - see README. The judge carries more weight
    because reference similarity punishes valid paraphrases; the semantic
    score stays in the blend as a deterministic, reference-anchored check.
    """
    return SEMANTIC_WEIGHT * semantic_similarity + JUDGE_WEIGHT * judge_score


def coerce_verdict(payload: Any) -> JudgeVerdict:
    """Validate a decoded judge payload, raising on anything malformed."""
    if not isinstance(payload, dict):
        raise ValueError(f"judge payload is {type(payload).__name__}, expected object")
    return JudgeVerdict.model_validate(payload)
