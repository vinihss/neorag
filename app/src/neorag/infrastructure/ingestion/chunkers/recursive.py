from langchain_text_splitters import RecursiveCharacterTextSplitter

from neorag.domain.entities import Document, Chunk
from neorag.infrastructure.ingestion.chunkers.base import Chunker
from neorag.infrastructure.ingestion.chunkers.factory import ChunkerFactory


@ChunkerFactory.register("recursive")
class RecursiveChunker(Chunker):
    def __init__(self) -> None:
        self._splitter: RecursiveCharacterTextSplitter | None = None

    def _get_splitter(
        self, chunk_size: int, overlap: float
    ) -> RecursiveCharacterTextSplitter:
        overlap_tokens = int(chunk_size * overlap)
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_tokens,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

    def chunk(
        self,
        documents: list[Document],
        chunk_size: int = 512,
        overlap: float = 0.15,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        splitter = self._get_splitter(chunk_size, overlap)

        for doc in documents:
            texts = splitter.split_text(doc.content)
            for i, text in enumerate(texts):
                chunks.append(
                    Chunk(
                        document_id=doc.id,
                        content=text,
                        source=doc.source,
                        page=doc.metadata.get("page"),
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "total_chunks": len(texts),
                        },
                    )
                )
        return chunks
