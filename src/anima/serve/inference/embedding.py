"""Embedding backends for retrieval.

- `BGEEmbedder`: the real backend, BAAI/bge-small-zh-v1.5 (512-dim, MIT).
  Vectors are L2-normalized so downstream similarity is a dot product. The
  retrieval-query instruction prefix recommended by the model card is applied
  only to queries, not to stored documents.
- `HashEmbedder`: a deterministic, dependency-free stand-in for unit tests so
  the request path can be exercised without loading a model. It is NOT for
  formal runs (real retrieval quality is only claimed with the real model).

Both satisfy the `Embedder` protocol in anima.persona.runtime.retrieval.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

from anima.persona.contracts.provenance import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION

BGE_MODEL_ID = EMBEDDING_MODEL_ID
BGE_MODEL_REVISION = EMBEDDING_MODEL_REVISION
BGE_DIM = 512
# bge-small-zh recommends this Chinese query instruction for asymmetric retrieval.
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return [1.0 / math.sqrt(len(vector))] * len(vector)
    return [v / norm for v in vector]


class HashEmbedder:
    """Deterministic hashing embedder for tests. Char 3-grams → dim buckets."""

    def __init__(self, dim: int = BGE_DIM) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        padded = f"^{text}$"
        for i in range(len(padded) - 2):
            gram = padded[i : i + 3]
            digest = hashlib.sha1(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        return _normalize(vec)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class BGEEmbedder:
    """Real bge-small-zh-v1.5 via sentence-transformers (lazy model load)."""

    def __init__(
        self,
        model_id: str = BGE_MODEL_ID,
        *,
        revision: str = BGE_MODEL_REVISION,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_id,
                revision=self.revision,
                device=self.device,
            )
        return self._model

    @property
    def dim(self) -> int:
        return int(self._ensure_model().get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return [[float(v) for v in row] for row in vectors]

    def embed_query(self, text: str) -> list[float]:
        [vector] = self.embed([BGE_QUERY_INSTRUCTION + text])
        return vector
