from typing import Protocol, runtime_checkable, Any

from neorag.domain.value_objects import ScoredChunk


@runtime_checkable
class RetrievalStrategy(Protocol):
    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]: ...
