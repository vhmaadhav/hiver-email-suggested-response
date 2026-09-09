"""Scoring arithmetic, the critical-error cap, and safe failure on bad judge output.

All LLM calls are mocked - these tests never touch the network.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from hiver_email.evaluation import SemanticScorer, human_validation, summarise
from hiver_email.generator import SuggestedReplyGenerator
from hiver_email.judge import RubricJudge
from hiver_email.llm import LLMError, parse_json_object
from hiver_email.retrieval import TfidfRetriever
from hiver_email.schemas import (
    CRITICAL_ERROR_SCORE_CAP,
    JUDGE_WEIGHTS,
    JudgeVerdict,
    ResponseScore,
    coerce_verdict,
    overall_score,
)

VALID = {
    "task_fulfillment": 4,
    "action_alignment": 4,
    "completeness": 4,
    "tone": 5,
    "critical_error": False,
    "acceptable": True,
    "reason": "Answers the request without inventing anything.",
}


def fake_client(response: str, gen_model="gen-x", judge_model="judge-y") -> MagicMock:
    client = MagicMock()
    client.gen_model = gen_model
    client.judge_model = judge_model
    client.complete.return_value = response
    return client


# --- weighting ------------------------------------------------------------
def test_judge_weights_sum_to_one():
    assert pytest.approx(sum(JUDGE_WEIGHTS.values())) == 1.0


def test_perfect_and_worst_rubric_scores_hit_the_scale_ends():
    best = JudgeVerdict(task_fulfillment=5, action_alignment=5, completeness=5, tone=5,
                        critical_error=False, acceptable=True)
    worst = JudgeVerdict(task_fulfillment=1, action_alignment=1, completeness=1, tone=1,
                         critical_error=False, acceptable=False)
    assert best.score_100() == pytest.approx(100.0)
    assert worst.score_100() == pytest.approx(0.0)


def test_judge_score_uses_the_documented_weights():
    v = JudgeVerdict(task_fulfillment=5, action_alignment=3, completeness=1, tone=4,
                     critical_error=False, acceptable=True)
    expected = (0.35 * 100 + 0.30 * 50 + 0.20 * 0 + 0.15 * 75)
    assert v.score_100() == pytest.approx(expected)


def test_task_fulfillment_moves_the_score_more_than_tone():
    base = dict(action_alignment=3, completeness=3, critical_error=False, acceptable=True)
    hi_task = JudgeVerdict(task_fulfillment=5, tone=3, **base).score_100()
    hi_tone = JudgeVerdict(task_fulfillment=3, tone=5, **base).score_100()
    assert hi_task > hi_tone


def test_overall_score_is_30_70():
    assert overall_score(100.0, 0.0) == pytest.approx(30.0)
    assert overall_score(0.0, 100.0) == pytest.approx(70.0)
    assert overall_score(50.0, 80.0) == pytest.approx(0.30 * 50 + 0.70 * 80)


# --- critical error cap ---------------------------------------------------
def test_critical_error_caps_the_judge_score():
    v = JudgeVerdict(task_fulfillment=5, action_alignment=5, completeness=5, tone=5,
                     critical_error=True, acceptable=False)
    assert v.score_100() == CRITICAL_ERROR_SCORE_CAP
    assert v.score_100() < 100.0


def test_critical_error_cap_never_raises_a_low_score():
    v = JudgeVerdict(task_fulfillment=1, action_alignment=1, completeness=1, tone=2,
                     critical_error=True, acceptable=False)
    assert v.score_100() < CRITICAL_ERROR_SCORE_CAP


# --- malformed judge output fails safely ----------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "I cannot evaluate this.", "[1, 2, 3]"])
def test_parse_json_object_rejects_non_objects(bad):
    with pytest.raises(ValueError):
        parse_json_object(bad)


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure, here it is: {"a": 1}') == {"a": 1}


def test_out_of_range_rubric_scores_are_rejected():
    with pytest.raises(ValidationError):
        coerce_verdict({**VALID, "task_fulfillment": 9})
    with pytest.raises(ValidationError):
        coerce_verdict({**VALID, "tone": 0})


def test_missing_field_is_rejected():
    payload = {k: v for k, v in VALID.items() if k != "acceptable"}
    with pytest.raises(ValidationError):
        coerce_verdict(payload)


def test_judge_raises_on_malformed_json_rather_than_defaulting():
    """A silent default would quietly fabricate a score - it must raise."""
    judge = RubricJudge(fake_client("the reply seems fine to me"))
    with pytest.raises(ValueError):
        judge.judge("incoming", "human reply", "candidate", [])


def test_judge_parses_a_valid_verdict():
    judge = RubricJudge(fake_client(json.dumps(VALID)))
    v = judge.judge("incoming", "human reply", "candidate", [])
    assert v.task_fulfillment == 4 and v.acceptable is True
    assert v.score_100() == pytest.approx(0.35 * 75 + 0.30 * 75 + 0.20 * 75 + 0.15 * 100)


def test_empty_candidate_is_scored_as_a_critical_failure_without_calling_the_llm():
    client = fake_client(json.dumps(VALID))
    v = RubricJudge(client).judge("incoming", "human reply", "   ", [])
    assert v.critical_error is True and v.acceptable is False
    assert v.score_100() == 0.0
    client.complete.assert_not_called()


def test_generator_records_llm_failure_instead_of_crashing(synthetic_pairs):
    client = fake_client("unused")
    client.complete.side_effect = LLMError("boom")
    gen = SuggestedReplyGenerator(TfidfRetriever(synthetic_pairs), client)
    result = gen.generate("invoice payment billing question")
    assert result.reply == ""
    assert "boom" in result.error
    assert result.retrieved, "retrieval still ran even though generation failed"


def test_generator_strips_markdown_fences(synthetic_pairs):
    client = fake_client("```\nHi there, confirming receipt.\n```")
    gen = SuggestedReplyGenerator(TfidfRetriever(synthetic_pairs), client)
    assert gen.generate("invoice payment billing").reply == "Hi there, confirming receipt."


# --- semantic signal ------------------------------------------------------
def test_semantic_scorer_ranks_related_above_unrelated():
    scorer = SemanticScorer()
    gen = ["The shipment leaves on Monday.", "The shipment leaves on Monday."]
    ref = ["Your shipment departs Monday morning.", "Please renew your gym membership."]
    related, unrelated = scorer.score_pairs(gen, ref)
    assert 0.0 <= unrelated < related <= 100.0


def test_semantic_scorer_gives_blank_output_zero():
    scorer = SemanticScorer()
    assert scorer.score_pairs(["   "], ["A real reference reply."])[0] == 0.0


def test_semantic_scorer_length_mismatch_raises():
    with pytest.raises(ValueError):
        SemanticScorer().score_pairs(["a"], ["a", "b"])


# --- aggregation ----------------------------------------------------------
def _row(system: str, judge: float, sem: float, ok: bool, err: str = "") -> ResponseScore:
    return ResponseScore(
        example_id=f"e{judge}{system}", system=system, incoming_email="i",
        human_reply="h", generated_reply="g", judge_score=judge,
        semantic_similarity=sem, acceptable=ok, critical_error=not ok,
        overall_score=overall_score(sem, judge), error=err,
    )


def test_summarise_excludes_failed_rows_from_means_but_counts_them():
    rows = [_row("main", 80, 60, True), _row("main", 40, 20, False),
            _row("main", 0, 0, False, err="judge_failed: timeout")]
    m = summarise(rows, "main")
    assert m.n == 2 and m.n_failed == 1
    assert m.mean_judge_score == pytest.approx(60.0)
    assert m.acceptable_rate == pytest.approx(0.5)


def test_summarise_of_an_unknown_system_is_empty():
    assert summarise([_row("main", 80, 60, True)], "baseline").n == 0


# --- human validation -----------------------------------------------------
def test_human_validation_returns_none_when_file_absent(tmp_path):
    assert human_validation([], tmp_path / "nope.csv") is None


def test_human_validation_reports_empty_without_inventing_labels(tmp_path):
    p = tmp_path / "hv.csv"
    p.write_text("example_id,human_score,human_acceptable\n", encoding="utf-8")
    assert human_validation([_row("main", 80, 60, True)], p)["status"] == "empty"


def test_human_validation_computes_spearman_and_agreement(tmp_path):
    rows = [
        ResponseScore(example_id="a", system="main", incoming_email="i", human_reply="h",
                      generated_reply="g", overall_score=90.0, acceptable=True),
        ResponseScore(example_id="b", system="main", incoming_email="i", human_reply="h",
                      generated_reply="g", overall_score=60.0, acceptable=True),
        ResponseScore(example_id="c", system="main", incoming_email="i", human_reply="h",
                      generated_reply="g", overall_score=20.0, acceptable=False),
    ]
    p = tmp_path / "hv.csv"
    p.write_text(
        "example_id,human_score,human_acceptable\na,95,true\nb,55,true\nc,10,false\n",
        encoding="utf-8",
    )
    out = human_validation(rows, p)
    assert out["status"] == "ok" and out["n_rated"] == 3
    assert out["spearman_rho"] == pytest.approx(1.0)
    assert out["acceptable_agreement"] == pytest.approx(1.0)


# --- leaked chain-of-thought guard ---------------------------------------
def test_generator_drops_leaked_chain_of_thought(synthetic_pairs):
    """A scratchpad must never be passed off as a suggested reply."""
    leaked = ("Here's a thinking process:\n1. **Analyze the User's Request:**\n"
              "   - I need to draft a reply...")
    gen = SuggestedReplyGenerator(TfidfRetriever(synthetic_pairs), fake_client(leaked))
    assert gen.generate("invoice payment billing").reply == ""


def test_generator_keeps_text_after_think_tag(synthetic_pairs):
    client = fake_client("<think>weighing the options</think>\nThanks - confirmed for Monday.")
    gen = SuggestedReplyGenerator(TfidfRetriever(synthetic_pairs), client)
    assert gen.generate("invoice payment billing").reply == "Thanks - confirmed for Monday."


def test_generator_keeps_a_normal_reply_untouched(synthetic_pairs):
    client = fake_client("Thanks for the note. I'll send the deck tonight.")
    gen = SuggestedReplyGenerator(TfidfRetriever(synthetic_pairs), client)
    assert gen.generate("send the deck").reply == "Thanks for the note. I'll send the deck tonight."
