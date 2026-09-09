"""TF-IDF retrieval over the historical corpus.

Deliberately simple: a single sparse matrix and a cosine dot product. On the
~12 k document corpus this indexes in a couple of seconds and uses tens of
megabytes, which is the right choice for CPU-only development and also the
choice that is easiest to explain and audit.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

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


# --------------------------------------------------------------------------
# Optional: dense and hybrid retrieval
#
# TF-IDF is the frozen default and produced the committed results. These are
# opt-in (--retriever dense|hybrid) because TF-IDF's ceiling is real: top-1
# cosine on this corpus is ~0.2, since a lexical match cannot see that
# "revised deck" and "updated presentation" are the same request.
#
# Still no FAISS and no vector database: the corpus embedding is a single
# (n, 384) float32 matrix - about 14 MB for 9.5 k emails - and search is one
# numpy matmul. At this scale an ANN index would add a dependency and cost
# recall without buying measurable speed.
# --------------------------------------------------------------------------
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseRetriever:
    """Embedding nearest-neighbour lookup over historical incoming emails."""

    def __init__(
        self,
        pairs: list[EmailPair],
        include_subject: bool = True,
        model_name: str = DEFAULT_DENSE_MODEL,
        cache_dir: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if not pairs:
            raise ValueError("cannot build a retriever over an empty corpus")
        self.pairs = list(pairs)
        self.include_subject = include_subject
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        texts = [self._doc_text(p) for p in self.pairs]
        self.matrix = self._embed_corpus(texts, cache_dir, batch_size)

    def _doc_text(self, pair: EmailPair) -> str:
        return pair.retrieval_text() if self.include_subject else pair.incoming_email

    def _cache_key(self, texts: list[str]) -> str:
        h = hashlib.sha1()
        h.update(self.model_name.encode())
        h.update(str(len(texts)).encode())
        for p in self.pairs:  # ids already content-addressed
            h.update(p.example_id.encode())
        return h.hexdigest()[:16]

    def _embed_corpus(
        self, texts: list[str], cache_dir: str | None, batch_size: int
    ) -> np.ndarray:
        """Encode once and cache: re-encoding 9.5 k emails per run is the
        single slowest thing in the pipeline on a CPU-only machine."""
        cache_path = None
        if cache_dir:
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)
            cache_path = directory / f"dense_{self._cache_key(texts)}.npy"
            if cache_path.exists():
                cached = np.load(cache_path)
                if cached.shape[0] == len(texts):
                    return cached
        matrix = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # so a dot product is cosine
            show_progress_bar=False,
        ).astype(np.float32)
        if cache_path is not None:
            np.save(cache_path, matrix)
        return matrix

    @property
    def size(self) -> int:
        return len(self.pairs)

    def scores(self, query: str) -> np.ndarray:
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return (self.matrix @ q.T).ravel()

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedExample]:
        if not query or not query.strip():
            return []
        return _top_k_from_scores(self.pairs, self.scores(query), top_k)


class HybridRetriever:
    """Convex blend of TF-IDF and dense cosine scores.

    Lexical and semantic retrieval fail differently: TF-IDF nails exact
    identifiers, contract names and ticket numbers that an embedding blurs;
    the embedding catches paraphrase that TF-IDF misses entirely. Blending the
    two normalised cosines keeps both, and `alpha` is reported in metrics.json
    rather than being treated as tuned - it was not fitted on anything.
    """

    def __init__(
        self,
        pairs: list[EmailPair],
        include_subject: bool = True,
        alpha: float = 0.5,
        model_name: str = DEFAULT_DENSE_MODEL,
        cache_dir: str | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self.sparse = TfidfRetriever(pairs, include_subject=include_subject)
        self.dense = DenseRetriever(
            pairs, include_subject=include_subject,
            model_name=model_name, cache_dir=cache_dir,
        )
        self.pairs = self.sparse.pairs

    @property
    def size(self) -> int:
        return len(self.pairs)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedExample]:
        if not query or not query.strip():
            return []
        sparse_scores = (self.sparse.matrix @ self.sparse.vectorizer
                         .transform([query]).T).toarray().ravel()
        combined = self.alpha * self.dense.scores(query) + (1 - self.alpha) * sparse_scores
        return _top_k_from_scores(self.pairs, combined, top_k)


def _top_k_from_scores(
    pairs: list[EmailPair], scores: np.ndarray, top_k: int
) -> list[RetrievedExample]:
    if scores.size == 0:
        return []
    k = min(top_k, scores.size)
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [
        RetrievedExample(
            example_id=pairs[i].example_id,
            incoming_email=pairs[i].incoming_email,
            human_reply=pairs[i].human_reply,
            similarity=float(scores[i]),
        )
        for i in top
        if scores[i] > 0.0
    ]


def build_retriever(
    pairs: list[EmailPair],
    kind: str = "tfidf",
    include_subject: bool = True,
    alpha: float = 0.5,
    cache_dir: str | None = None,
):
    """Factory so scripts can switch retrievers with one flag."""
    kind = kind.lower()
    if kind == "tfidf":
        return TfidfRetriever(pairs, include_subject=include_subject)
    if kind == "dense":
        return DenseRetriever(pairs, include_subject=include_subject,
                              cache_dir=cache_dir)
    if kind == "hybrid":
        return HybridRetriever(pairs, include_subject=include_subject,
                               alpha=alpha, cache_dir=cache_dir)
    raise ValueError(f"unknown retriever {kind!r}; use tfidf, dense or hybrid")
