from typing import Protocol, runtime_checkable

from neorag.domain.entities import Document, Chunk


@runtime_checkable
class Chunker(Protocol):
    def chunk(
        self,
        documents: list[Document],
        chunk_size: int = 512,
        overlap: float = 0.15,
    ) -> list[Chunk]: ...
