"""Dataset loading, normalisation and the deterministic 80/20 split.

The split is the load-bearing part of the evaluation story: the 20% held-out
pool must never appear in the retrieval corpus, otherwise the system can
retrieve the very reply it is being scored against.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .schemas import EmailPair

DEFAULT_SEED = 13
DEFAULT_TEST_FRACTION = 0.20

# The Kaggle dataset ships slightly different capitalisations across versions,
# so we resolve columns by a list of candidates rather than assuming one name.
COLUMN_CANDIDATES: dict[str, list[str]] = {
    "incoming_email": ["EmailSend", "email_send", "emailsend", "Email_Send", "body_send"],
    "human_reply": ["EmailReply", "email_reply", "emailreply", "Email_Reply", "body_reply"],
    "subject_send": ["SubjectSend", "subject_send", "subjectsend", "Subject_Send"],
    "subject_reply": ["SubjectReply", "subject_reply", "subjectreply", "Subject_Reply"],
    "sender": ["From", "from", "sender", "From_Send"],
    "recipient": ["To", "to", "recipient", "To_Send"],
    "date_send": ["DateSend", "date_send", "datesend", "Date_Send", "Date"],
}

MIN_EMAIL_CHARS = 40
MIN_REPLY_CHARS = 20
MAX_EMAIL_CHARS = 4000  # keeps prompts (and 16 GB RAM) comfortable


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return Path(os.environ.get("HIVER_DATA_DIR", project_root() / "data"))


def find_dataset_csv(data_dir: Path | None = None) -> Path:
    """Locate the dataset CSV inside the data directory."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    candidates = sorted(data_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found under {data_dir}.\n"
            "Run:  uv run python scripts/fetch_data.py\n"
            "or download 'oanannv/enron-email-reply-dataset' from Kaggle manually "
            f"and place the CSV in {data_dir}."
        )
    # Prefer the biggest CSV - the dataset file, not an auxiliary index.
    return max(candidates, key=lambda p: p.stat().st_size)


def _resolve_column(df: pd.DataFrame, logical: str) -> str | None:
    lowered = {c.lower().strip(): c for c in df.columns}
    for candidate in COLUMN_CANDIDATES[logical]:
        hit = lowered.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def _clean(text: object) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    # Collapse runaway blank lines but keep paragraph structure.
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s[:MAX_EMAIL_CHARS]


def stable_id(incoming: str, reply: str) -> str:
    """Content-addressed id, so ids survive re-downloads and row reordering."""
    digest = hashlib.sha1(f"{incoming}\x00{reply}".encode("utf-8", "ignore")).hexdigest()
    return digest[:16]


def load_pairs(csv_path: Path | str | None = None) -> list[EmailPair]:
    """Load and normalise the dataset into EmailPair records."""
    path = Path(csv_path) if csv_path else find_dataset_csv()
    df = pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip")

    col_in = _resolve_column(df, "incoming_email")
    col_reply = _resolve_column(df, "human_reply")
    if col_in is None or col_reply is None:
        raise ValueError(
            f"Could not find incoming/reply columns in {path}. "
            f"Columns present: {list(df.columns)}"
        )
    optional = {k: _resolve_column(df, k) for k in
                ("subject_send", "subject_reply", "sender", "recipient", "date_send")}

    pairs: list[EmailPair] = []
    seen: set[str] = set()
    for row in df.itertuples(index=False):
        record = dict(zip(df.columns, row))
        incoming = _clean(record.get(col_in))
        reply = _clean(record.get(col_reply))
        if len(incoming) < MIN_EMAIL_CHARS or len(reply) < MIN_REPLY_CHARS:
            continue
        eid = stable_id(incoming, reply)
        if eid in seen:  # exact duplicate pair
            continue
        seen.add(eid)
        pairs.append(
            EmailPair(
                example_id=eid,
                incoming_email=incoming,
                human_reply=reply,
                subject_send=_clean(record.get(optional["subject_send"])),
                subject_reply=_clean(record.get(optional["subject_reply"])),
                sender=_clean(record.get(optional["sender"]))[:200],
                recipient=_clean(record.get(optional["recipient"]))[:200],
                date_send=_clean(record.get(optional["date_send"]))[:64],
            )
        )
    if not pairs:
        raise ValueError(f"No usable email/reply pairs survived cleaning in {path}")
    return pairs


def split_pairs(
    pairs: list[EmailPair],
    seed: int = DEFAULT_SEED,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[list[EmailPair], list[EmailPair]]:
    """Deterministic 80/20 split into (retrieval corpus, held-out pool).

    Deterministic given (pairs, seed): a seeded numpy permutation over ids
    sorted for stability, so the split does not depend on CSV row order.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    ordered = sorted(pairs, key=lambda p: p.example_id)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ordered))
    n_test = max(1, int(round(len(ordered) * test_fraction)))
    test_idx = set(perm[:n_test].tolist())
    corpus = [p for i, p in enumerate(ordered) if i not in test_idx]
    heldout = [p for i, p in enumerate(ordered) if i in test_idx]
    return corpus, heldout


def sample_heldout(
    heldout: list[EmailPair], n: int, seed: int = DEFAULT_SEED
) -> list[EmailPair]:
    """Deterministic sub-sample of the held-out pool for a cheap eval run."""
    if n >= len(heldout):
        return list(heldout)
    ordered = sorted(heldout, key=lambda p: p.example_id)
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(ordered))[:n]
    return [ordered[i] for i in sorted(idx.tolist())]
