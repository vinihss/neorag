from typing import Protocol, runtime_checkable

from neorag.domain.entities import Document


@runtime_checkable
class DocumentLoader(Protocol):
    async def load(self, source: str) -> list[Document]: ...
