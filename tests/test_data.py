"""The split is the backbone of the evaluation claim, so it is tested hardest."""

from __future__ import annotations

import pandas as pd
import pytest

from hiver_email.data import load_pairs, sample_heldout, split_pairs, stable_id


def test_split_is_deterministic(synthetic_pairs):
    a_corpus, a_held = split_pairs(synthetic_pairs, seed=13)
    b_corpus, b_held = split_pairs(synthetic_pairs, seed=13)
    assert [p.example_id for p in a_corpus] == [p.example_id for p in b_corpus]
    assert [p.example_id for p in a_held] == [p.example_id for p in b_held]


def test_split_depends_on_seed(synthetic_pairs):
    _, held_a = split_pairs(synthetic_pairs, seed=13)
    _, held_b = split_pairs(synthetic_pairs, seed=99)
    assert {p.example_id for p in held_a} != {p.example_id for p in held_b}


def test_split_is_order_independent(synthetic_pairs):
    """Shuffling the input CSV must not change who lands in the held-out pool."""
    shuffled = list(reversed(synthetic_pairs))
    _, held_a = split_pairs(synthetic_pairs, seed=13)
    _, held_b = split_pairs(shuffled, seed=13)
    assert {p.example_id for p in held_a} == {p.example_id for p in held_b}


def test_split_is_a_partition_with_no_leakage(synthetic_pairs):
    corpus, held = split_pairs(synthetic_pairs, seed=13)
    corpus_ids = {p.example_id for p in corpus}
    held_ids = {p.example_id for p in held}
    assert corpus_ids & held_ids == set(), "held-out examples leaked into the corpus"
    assert corpus_ids | held_ids == {p.example_id for p in synthetic_pairs}
    assert len(corpus) + len(held) == len(synthetic_pairs)


def test_split_respects_test_fraction(synthetic_pairs):
    _, held = split_pairs(synthetic_pairs, seed=13, test_fraction=0.20)
    assert len(held) == round(len(synthetic_pairs) * 0.20)


def test_split_rejects_bad_fraction(synthetic_pairs):
    with pytest.raises(ValueError):
        split_pairs(synthetic_pairs, test_fraction=1.5)


def test_sample_heldout_is_deterministic_and_bounded(synthetic_pairs):
    _, held = split_pairs(synthetic_pairs, seed=13)
    a = sample_heldout(held, 4, seed=13)
    b = sample_heldout(held, 4, seed=13)
    assert [p.example_id for p in a] == [p.example_id for p in b]
    assert len(a) == 4
    assert set(p.example_id for p in a) <= set(p.example_id for p in held)
    # Asking for more than exists returns everything rather than raising.
    assert len(sample_heldout(held, 10_000, seed=13)) == len(held)


def test_stable_id_is_content_addressed():
    assert stable_id("a", "b") == stable_id("a", "b")
    assert stable_id("a", "b") != stable_id("b", "a")


def test_load_pairs_normalises_and_filters(tmp_path):
    """Real column names, short rows dropped, duplicates collapsed."""
    csv = tmp_path / "d.csv"
    pd.DataFrame(
        {
            "EmailSend": [
                "Please confirm the delivery schedule for next week's shipment.",
                "Please confirm the delivery schedule for next week's shipment.",
                "too short",
                "Could you review the attached contract and share your comments?",
            ],
            "EmailReply": [
                "Confirmed, the shipment leaves Monday.",
                "Confirmed, the shipment leaves Monday.",
                "Sure thing, will do that today.",
                "no",  # reply too short
            ],
            "SubjectSend": ["Shipment", "Shipment", "x", "Contract"],
            "SubjectReply": ["Re: Shipment", "Re: Shipment", "Re: x", "Re: Contract"],
            "From": ["a@x.com"] * 4,
            "To": ["b@x.com"] * 4,
            "DateSend": ["2001-01-01"] * 4,
        }
    ).to_csv(csv, index=False)

    pairs = load_pairs(csv)
    assert len(pairs) == 1  # dupe collapsed, short email dropped, short reply dropped
    assert pairs[0].subject_send == "Shipment"
    assert "Shipment" in pairs[0].retrieval_text()


def test_load_pairs_errors_on_unknown_schema(tmp_path):
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": ["a"], "bar": ["b"]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Could not find incoming/reply columns"):
        load_pairs(csv)
