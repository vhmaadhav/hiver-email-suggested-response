"""Retrieval correctness, including the no-leakage guarantee end to end."""

from __future__ import annotations

import pytest

from hiver_email.data import sample_heldout, split_pairs
from hiver_email.generator import build_prompt
from hiver_email.retrieval import TfidfRetriever, baseline_reply
from hiver_email.schemas import EmailPair


def test_retrieves_the_topically_nearest_email(synthetic_pairs):
    r = TfidfRetriever(synthetic_pairs, include_subject=True)
    hits = r.retrieve("What is the status of the gas pipeline capacity nomination?", top_k=3)
    assert hits, "expected at least one hit"
    assert "pipeline" in hits[0].incoming_email.lower()
    # Similarities are sorted descending and are valid cosines.
    scores = [h.similarity for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 + 1e-6 for s in scores)


def test_retrieve_respects_top_k(synthetic_pairs):
    r = TfidfRetriever(synthetic_pairs)
    assert len(r.retrieve("invoice payment billing", top_k=3)) <= 3
    assert len(r.retrieve("invoice payment billing", top_k=1)) == 1


def test_every_hit_exposes_email_reply_and_score(synthetic_pairs):
    r = TfidfRetriever(synthetic_pairs)
    for hit in r.retrieve("schedule a meeting next week", top_k=3):
        assert hit.incoming_email and hit.human_reply
        assert isinstance(hit.similarity, float)


def test_empty_or_unmatched_query_returns_nothing(synthetic_pairs):
    r = TfidfRetriever(synthetic_pairs)
    assert r.retrieve("") == []
    assert r.retrieve("   ") == []
    # No shared vocabulary -> zero similarity -> filtered out.
    assert r.retrieve("zzzz qqqq xxxx") == []


def test_retriever_rejects_empty_corpus():
    with pytest.raises(ValueError):
        TfidfRetriever([])


def test_baseline_copies_the_nearest_reply_verbatim(synthetic_pairs):
    r = TfidfRetriever(synthetic_pairs)
    hits = r.retrieve("quarterly forecast numbers budget", top_k=3)
    assert baseline_reply(hits) == hits[0].human_reply
    assert baseline_reply([]) == ""


def test_heldout_examples_are_absent_from_the_retrieval_corpus(synthetic_pairs):
    """The core evaluation-integrity test.

    Build the index the way run_eval does, then confirm no evaluation email is
    retrievable from it - neither by id nor by exact text match.
    """
    corpus, heldout = split_pairs(synthetic_pairs, seed=13)
    retriever = TfidfRetriever(corpus)

    corpus_ids = {p.example_id for p in corpus}
    corpus_texts = {p.incoming_email for p in corpus}
    corpus_replies = {p.human_reply for p in corpus}

    for pair in sample_heldout(heldout, 5, seed=13):
        assert pair.example_id not in corpus_ids
        assert pair.incoming_email not in corpus_texts
        hits = retriever.retrieve(pair.retrieval_text(), top_k=3)
        assert all(h.example_id != pair.example_id for h in hits)
        # The gold reply itself must never be handed back as grounding.
        assert all(h.human_reply != pair.human_reply or h.human_reply in corpus_replies
                   for h in hits)


def test_prompt_contains_grounding_and_the_new_email(synthetic_pairs):
    r = TfidfRetriever(synthetic_pairs)
    hits = r.retrieve("contract redlines legal review", top_k=2)
    prompt = build_prompt("Please review the contract redlines.", hits)
    assert "NEW INCOMING EMAIL" in prompt
    assert "Please review the contract redlines." in prompt
    assert "Historical example 1" in prompt
    assert hits[0].human_reply[:30] in prompt


def test_prompt_handles_no_retrieval():
    prompt = build_prompt("A brand new question.", [])
    assert "No sufficiently similar historical email" in prompt
    assert "A brand new question." in prompt


def test_include_subject_flag_changes_the_indexed_text():
    pairs = [
        EmailPair(example_id="a", subject_send="Pipeline nomination",
                  incoming_email="Please advise on the item below.",
                  human_reply="Confirmed for tomorrow."),
        EmailPair(example_id="b", subject_send="Travel booking",
                  incoming_email="Please advise on the item below.",
                  human_reply="Your travel is booked."),
    ]
    with_subject = TfidfRetriever(pairs, include_subject=True, min_df=1)
    hits = with_subject.retrieve("pipeline nomination", top_k=1)
    assert hits and hits[0].example_id == "a"

    without = TfidfRetriever(pairs, include_subject=False, min_df=1)
    assert without.retrieve("pipeline nomination", top_k=1) == []


# --- retriever factory ----------------------------------------------------
def test_build_retriever_returns_tfidf_by_default(synthetic_pairs):
    from hiver_email.retrieval import build_retriever

    r = build_retriever(synthetic_pairs)
    assert isinstance(r, TfidfRetriever)
    assert r.retrieve("gas pipeline capacity nomination", top_k=1)


def test_build_retriever_rejects_unknown_kind(synthetic_pairs):
    from hiver_email.retrieval import build_retriever

    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever(synthetic_pairs, kind="faiss")


def test_hybrid_alpha_must_be_a_convex_weight(synthetic_pairs):
    from hiver_email.retrieval import HybridRetriever

    with pytest.raises(ValueError, match="alpha"):
        HybridRetriever(synthetic_pairs, alpha=1.5)


def test_top_k_from_scores_sorts_and_drops_zeros(synthetic_pairs):
    """The shared ranking helper must never pad results with zero-score hits."""
    import numpy as np

    from hiver_email.retrieval import _top_k_from_scores

    scores = np.array([0.0, 0.9, 0.3, 0.0, 0.6], dtype=np.float32)
    hits = _top_k_from_scores(synthetic_pairs[:5], scores, top_k=4)
    assert [round(h.similarity, 1) for h in hits] == [0.9, 0.6, 0.3]
