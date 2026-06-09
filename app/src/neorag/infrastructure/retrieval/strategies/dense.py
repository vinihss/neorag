from typing import Any

from neorag.domain.ports import Embedder, VectorStore
from neorag.domain.value_objects import ScoredChunk
from neorag.infrastructure.retrieval.strategies.base import RetrievalStrategy
from neorag.infrastructure.retrieval.strategies.factory import RetrievalStrategyFactory


@RetrievalStrategyFactory.register("dense")
class DenseRetrievalStrategy(RetrievalStrategy):
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        collection_name: str = "documents",
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._collection_name = collection_name

    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        query_vector = await self._embedder.embed_query(query)
        results = await self._vector_store.search(
            collection=self._collection_name,
            query_vector=query_vector,
            k=k,
            filter_=filter_,
        )
        return results
