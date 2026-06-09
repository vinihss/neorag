from typing import Any

from rank_bm25 import BM25Okapi

from neorag.domain.value_objects import ScoredChunk
from neorag.infrastructure.retrieval.strategies.base import RetrievalStrategy
from neorag.infrastructure.retrieval.strategies.factory import RetrievalStrategyFactory


@RetrievalStrategyFactory.register("sparse")
class SparseRetrievalStrategy(RetrievalStrategy):
    def __init__(self, texts: list[str] | None = None, ids: list[str] | None = None) -> None:
        self._texts: list[str] = texts or []
        self._ids: list[str] = ids or []
        self._corpus: list[list[str]] = [self._tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def index(self, texts: list[str], ids: list[str]) -> None:
        self._texts = texts
        self._ids = ids
        self._corpus = [self._tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus)

    def add(self, text: str, id_: str) -> None:
        self._texts.append(text)
        self._ids.append(id_)
        self._corpus.append(self._tokenize(text))
        self._bm25 = BM25Okapi(self._corpus)

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        if not self._bm25 or not self._texts:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for idx, score in indexed[:k]:
            results.append(
                ScoredChunk(
                    chunk_id=self._ids[idx] if idx < len(self._ids) else "",
                    score=float(score),
                    text=self._texts[idx] if idx < len(self._texts) else "",
                )
            )
        return results
