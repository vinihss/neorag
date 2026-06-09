from neorag.config import RetrievalConfig
from neorag.domain.entities import Query
from neorag.domain.ports import Embedder, VectorStore
from neorag.domain.value_objects import ScoredChunk
from neorag.infrastructure.retrieval.rerankers.base import Reranker
from neorag.infrastructure.retrieval.strategies.dense import DenseRetrievalStrategy
from neorag.infrastructure.retrieval.strategies.hybrid import HybridRetrievalStrategy
from neorag.infrastructure.retrieval.strategies.sparse import SparseRetrievalStrategy


class RetrievalService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        config: RetrievalConfig,
        reranker: Reranker | None = None,
        collection_name: str = "documents",
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._config = config
        self._reranker = reranker
        self._collection_name = collection_name
        self._dense = DenseRetrievalStrategy(
            embedder=embedder,
            vector_store=vector_store,
            collection_name=collection_name,
        )
        self._sparse: SparseRetrievalStrategy | None = None
        self._hybrid: HybridRetrievalStrategy | None = None

    def _build_sparse(self) -> SparseRetrievalStrategy | None:
        if self._sparse is not None:
            return self._sparse
        return None

    async def retrieve(self, query: Query) -> list[ScoredChunk]:
        filter_ = query.filter.copy()

        if query.user_id:
            filter_["_should"] = [
                {"user_id": query.user_id},
                {"is_public": True},
            ]

        alpha = self._config.alpha

        if alpha <= 0.0:
            results = await self._dense.retrieve(
                query.text, k=query.top_k, filter_=filter_
            )
        elif alpha >= 1.0 and self._sparse:
            results = await self._sparse.retrieve(
                query.text, k=query.top_k, filter_=filter_
            )
        elif self._sparse:
            hybrid = HybridRetrievalStrategy(
                dense=self._dense, sparse=self._sparse, alpha=alpha
            )
            results = await hybrid.retrieve(
                query.text, k=query.top_k, filter_=filter_
            )
        else:
            results = await self._dense.retrieve(
                query.text, k=query.top_k, filter_=filter_
            )

        if self._reranker and results:
            results = await self._reranker.rerank(
                query.text, results, top_k=query.top_k
            )

        results = [r for r in results if r.score >= self._config.min_score]
        return results[: query.top_k]
