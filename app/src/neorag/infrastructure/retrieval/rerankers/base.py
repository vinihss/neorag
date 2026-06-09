from typing import Protocol, runtime_checkable

from neorag.domain.value_objects import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]: ...
