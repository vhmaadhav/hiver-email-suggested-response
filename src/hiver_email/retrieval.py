"""TF-IDF retrieval over the historical corpus.

Deliberately simple: a single sparse matrix and a cosine dot product. On the
~12 k document corpus this indexes in a couple of seconds and uses tens of
megabytes, which is the right choice for CPU-only development and also the
choice that is easiest to explain and audit.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .schemas import EmailPair, RetrievedExample

DEFAULT_TOP_K = 3


class TfidfRetriever:
    """Nearest-neighbour lookup over historical *incoming* emails."""

    def __init__(
        self,
        pairs: list[EmailPair],
        include_subject: bool = True,
        max_features: int = 50_000,
        min_df: int = 2,
    ) -> None:
        if not pairs:
            raise ValueError("cannot build a retriever over an empty corpus")
        self.pairs = list(pairs)
        self.include_subject = include_subject
        texts = [self._doc_text(p) for p in self.pairs]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=min_df if len(texts) > 50 else 1,
            sublinear_tf=True,
            dtype=np.float32,  # halves index memory vs float64
        )
        # TF-IDF rows are L2-normalised by default, so a dot product is cosine.
        self.matrix = self.vectorizer.fit_transform(texts)

    def _doc_text(self, pair: EmailPair) -> str:
        return pair.retrieval_text() if self.include_subject else pair.incoming_email

    @property
    def size(self) -> int:
        return len(self.pairs)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedExample]:
        """Return the top_k most similar historical pairs for a new email."""
        if not query or not query.strip():
            return []
        q = self.vectorizer.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()
        if sims.size == 0:
            return []
        k = min(top_k, sims.size)
        # argpartition then sort the short list: O(n) instead of a full sort.
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [
            RetrievedExample(
                example_id=self.pairs[i].example_id,
                incoming_email=self.pairs[i].incoming_email,
                human_reply=self.pairs[i].human_reply,
                similarity=float(sims[i]),
            )
            for i in top
            if sims[i] > 0.0
        ]


def baseline_reply(retrieved: list[RetrievedExample]) -> str:
    """Retrieval-only baseline: copy the nearest historical human reply verbatim."""
    return retrieved[0].human_reply if retrieved else ""
