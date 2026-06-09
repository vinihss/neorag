from typing import Any

from neorag.domain.value_objects import ScoredChunk
from neorag.infrastructure.retrieval.strategies.base import RetrievalStrategy
from neorag.infrastructure.retrieval.strategies.factory import RetrievalStrategyFactory
from neorag.infrastructure.retrieval.strategies.dense import DenseRetrievalStrategy
from neorag.infrastructure.retrieval.strategies.sparse import SparseRetrievalStrategy


@RetrievalStrategyFactory.register("hybrid")
class HybridRetrievalStrategy(RetrievalStrategy):
    def __init__(
        self,
        dense: DenseRetrievalStrategy,
        sparse: SparseRetrievalStrategy,
        alpha: float = 0.5,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._alpha = alpha

    def _normalize(
        self, results: list[ScoredChunk]
    ) -> dict[str, float]:
        if not results:
            return {}
        scores = [r.score for r in results]
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 1e-9:
            return {r.chunk_id: 1.0 for r in results}
        return {r.chunk_id: (r.score - min_s) / (max_s - min_s) for r in results}

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        dense_k = k * 2 if k > 0 else 20
        dense_results = await self._dense.retrieve(query, k=dense_k, filter_=filter_)
        sparse_results = await self._sparse.retrieve(query, k=dense_k, filter_=filter_)

        dense_norm = self._normalize(dense_results)
        sparse_norm = self._normalize(sparse_results)

        combined: dict[str, tuple[float, ScoredChunk]] = {}

        for r in dense_results:
            score = self._alpha * dense_norm.get(r.chunk_id, 0.0)
            combined[r.chunk_id] = (score, r)

        for r in sparse_results:
            sparse_score = (1 - self._alpha) * sparse_norm.get(r.chunk_id, 0.0)
            if r.chunk_id in combined:
                existing_score, existing = combined[r.chunk_id]
                combined[r.chunk_id] = (
                    existing_score + sparse_score,
                    existing,
                )
            else:
                combined[r.chunk_id] = (sparse_score, r)

        sorted_results = sorted(
            combined.values(), key=lambda x: x[0], reverse=True
        )
        return [r for score, r in sorted_results[:k]]
