import pytest

from neorag.domain.entities import Document
from neorag.infrastructure.ingestion.chunkers.factory import ChunkerFactory
from neorag.infrastructure.ingestion.chunkers.recursive import RecursiveChunker


class TestChunkerFactory:
    def test_registered_chunkers(self):
        assert "recursive" in ChunkerFactory._registry

    def test_create_recursive(self):
        chunker = ChunkerFactory.create("recursive")
        assert isinstance(chunker, RecursiveChunker)

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown chunker"):
            ChunkerFactory.create("unknown")


class TestRecursiveChunker:
    def test_empty_documents(self):
        chunker = RecursiveChunker()
        chunks = chunker.chunk([])
        assert chunks == []

    def test_small_document(self):
        chunker = RecursiveChunker()
        doc = Document(content="Small text.", source="test.md")
        chunks = chunker.chunk([doc], chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].content == "Small text."
        assert chunks[0].document_id == doc.id
        assert chunks[0].source == "test.md"

    def test_large_document_splits(self):
        chunker = RecursiveChunker()
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\n"
        doc = Document(content=content, source="test.md")
        chunks = chunker.chunk([doc], chunk_size=20, overlap=0)
        assert len(chunks) > 1

    def test_metadata_propagation(self):
        chunker = RecursiveChunker()
        doc = Document(
            content="Test content here.",
            source="doc.pdf",
            metadata={"page": 1, "author": "test"},
        )
        chunks = chunker.chunk([doc])
        assert chunks[0].source == "doc.pdf"
        assert chunks[0].metadata.get("page") == 1
        assert chunks[0].metadata.get("author") == "test"

    def test_chunk_index_tracking(self):
        chunker = RecursiveChunker()
        content = "A.\n\nB.\n\nC.\n\nD.\n\nE.\n\n"
        doc = Document(content=content, source="test.md")
        chunks = chunker.chunk([doc], chunk_size=10, overlap=0)
        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_index"] == i
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_multiple_documents(self):
        chunker = RecursiveChunker()
        docs = [
            Document(content="First doc.", source="a.md"),
            Document(content="Second doc.", source="b.md"),
        ]
        chunks = chunker.chunk(docs)
        assert len(chunks) == 2
        assert chunks[0].document_id == docs[0].id
        assert chunks[1].document_id == docs[1].id
