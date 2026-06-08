import pytest
from typing import AsyncIterator

from neorag.domain.ports import (
    Embedder,
    VectorStore,
    DocumentLoader,
    Chunker,
    Reranker,
    LLM,
    SessionRepository,
)
from neorag.domain.entities import Document, Chunk, Message
from neorag.domain.value_objects import ScoredChunk


class MockEmbedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 4 for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * 4

    @property
    def dimensions(self) -> int:
        return 4


class MockVectorStore:
    def __init__(self):
        self.points = []

    async def add(self, collection: str, points: list[dict]) -> None:
        self.points.extend(points)

    async def search(
        self, collection: str, query_vector: list[float], k: int = 10, filter_: dict | None = None
    ) -> list[ScoredChunk]:
        return [ScoredChunk(chunk_id="c1", score=0.9)]

    async def delete(self, collection: str, point_ids: list[str]) -> None:
        self.points = [p for p in self.points if p.get("id") not in point_ids]

    async def collection_exists(self, collection: str) -> bool:
        return True

    async def create_collection(self, collection: str, vector_size: int) -> None:
        pass


class MockLoader:
    async def load(self, source: str) -> list[Document]:
        return [Document(content="test", source=source)]


class MockChunker:
    def chunk(
        self, documents: list[Document], chunk_size: int = 512, overlap: float = 0.15
    ) -> list[Chunk]:
        return [
            Chunk(document_id=doc.id, content=doc.content[:chunk_size])
            for doc in documents
        ]


class MockReranker:
    async def rerank(
        self, query: str, candidates: list[ScoredChunk], top_k: int = 5
    ) -> list[ScoredChunk]:
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]


class MockLLM:
    async def generate(self, prompt: str, **kwargs) -> str:
        return f"Resposta para: {prompt[:50]}"

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        for char in "resposta":
            yield char


class MockSessionRepo:
    def __init__(self):
        self.sessions = {}

    async def get(self, session_id: str):
        return self.sessions.get(session_id)

    async def save(self, session) -> None:
        self.sessions[session.id] = session

    async def add_message(self, session_id: str, message: Message) -> None:
        if session := self.sessions.get(session_id):
            session.add_message(message)

    async def get_history(self, session_id: str, limit: int = 10):
        session = self.sessions.get(session_id)
        return session.last_messages(limit) if session else []

    async def list_sessions(self, user_id: str | None = None):
        return list(self.sessions.values())


class TestEmbedderProtocol:
    def test_mock_implements_protocol(self):
        assert isinstance(MockEmbedder(), Embedder)

    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self):
        embedder = MockEmbedder()
        result = await embedder.embed(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 4

    def test_dimensions_property(self):
        embedder = MockEmbedder()
        assert embedder.dimensions == 4


class TestVectorStoreProtocol:
    def test_mock_implements_protocol(self):
        assert isinstance(MockVectorStore(), VectorStore)

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        store = MockVectorStore()
        await store.add("test", [{"id": "1", "vector": [0.1, 0.2]}])
        results = await store.search("test", [0.1, 0.2])
        assert len(results) > 0


class TestLLMProtocol:
    def test_mock_implements_protocol(self):
        assert isinstance(MockLLM(), LLM)

    @pytest.mark.asyncio
    async def test_generate(self):
        llm = MockLLM()
        result = await llm.generate("pergunta")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        llm = MockLLM()
        chunks = []
        async for chunk in llm.generate_stream("test"):
            chunks.append(chunk)
        assert len(chunks) > 0


class TestRerankerProtocol:
    def test_mock_implements_protocol(self):
        assert isinstance(MockReranker(), Reranker)


class TestLoaderProtocol:
    def test_mock_implements_protocol(self):
        assert isinstance(MockLoader(), DocumentLoader)


class TestChunkerProtocol:
    def test_mock_implements_protocol(self):
        assert isinstance(MockChunker(), Chunker)


class TestSessionRepoProtocol:
    def test_mock_implements_protocol(self):
        assert isinstance(MockSessionRepo(), SessionRepository)
