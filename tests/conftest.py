from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiver_email.schemas import EmailPair  # noqa: E402


def make_pair(i: int, incoming: str, reply: str, subject: str = "") -> EmailPair:
    return EmailPair(
        example_id=f"id{i:04d}",
        subject_send=subject,
        incoming_email=incoming,
        human_reply=reply,
    )


@pytest.fixture
def synthetic_pairs() -> list[EmailPair]:
    """A small, fully deterministic corpus - no network, no dataset needed."""
    topics = [
        ("invoice payment overdue account billing", "We have processed the invoice."),
        ("schedule a meeting next week calendar", "Tuesday at 10am works for me."),
        ("send the revised slide deck presentation", "The deck is attached."),
        ("gas pipeline capacity nomination volume", "Nomination confirmed for tomorrow."),
        ("contract redlines legal review counsel", "Legal has the redlines now."),
        ("server outage downtime incident report", "The outage is resolved."),
        ("quarterly forecast numbers budget", "Forecast is in the shared folder."),
        ("travel booking flight hotel expense", "Your travel is booked."),
    ]
    pairs = []
    for i in range(40):
        topic, reply = topics[i % len(topics)]
        pairs.append(
            make_pair(i, f"Hello, regarding {topic}, please advise on item {i}.",
                      f"{reply} Reference {i}.", subject=f"Re: {topic}")
        )
    return pairs
