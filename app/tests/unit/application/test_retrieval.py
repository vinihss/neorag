import pytest
from unittest.mock import AsyncMock, MagicMock

from neorag.application.retrieval import RetrievalService
from neorag.config import RetrievalConfig
from neorag.domain.entities import Query
from neorag.domain.ports import Embedder, VectorStore
from neorag.domain.value_objects import ScoredChunk


@pytest.fixture
def mock_embedder():
    m = MagicMock(spec=Embedder)
    m.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return m


@pytest.fixture
def mock_vector_store():
    m = MagicMock(spec=VectorStore)
    m.search = AsyncMock(
        return_value=[
            ScoredChunk(chunk_id="c1", score=0.9, text="doc1", source="s1"),
            ScoredChunk(chunk_id="c2", score=0.7, text="doc2", source="s1"),
            ScoredChunk(chunk_id="c3", score=0.5, text="doc3", source="s2"),
        ]
    )
    return m


@pytest.fixture
def config():
    return RetrievalConfig(
        alpha=0.0,
        top_k_retrieve=20,
        top_k_rerank=5,
        min_score=0.0,
    )


class TestRetrievalService:
    @pytest.mark.asyncio
    async def test_retrieve_basic(self, mock_embedder, mock_vector_store, config):
        service = RetrievalService(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config,
        )

        query = Query(text="test query", top_k=2)
        results = await service.retrieve(query)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_retrieve_with_min_score_filter(self, mock_embedder, mock_vector_store):
        config = RetrievalConfig(alpha=0.0, min_score=0.8)
        service = RetrievalService(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config,
        )

        query = Query(text="test", top_k=5)
        results = await service.retrieve(query)

        assert len(results) == 1
        assert results[0].score >= 0.8

    @pytest.mark.asyncio
    async def test_retrieve_with_user_filter(self, mock_embedder, mock_vector_store, config):
        service = RetrievalService(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config,
        )

        query = Query(text="test", top_k=5, user_id="user123")
        await service.retrieve(query)

        _call_kwargs = mock_vector_store.search.call_args.kwargs
        assert "_should" in _call_kwargs["filter_"]
        assert {"user_id": "user123"} in _call_kwargs["filter_"]["_should"]
        assert {"is_public": True} in _call_kwargs["filter_"]["_should"]

    @pytest.mark.asyncio
    async def test_retrieve_with_custom_filter(self, mock_embedder, mock_vector_store, config):
        service = RetrievalService(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            config=config,
        )

        query = Query(text="test", top_k=5, filter={"doc_type": "pdf"})
        await service.retrieve(query)

        _call_kwargs = mock_vector_store.search.call_args.kwargs
        assert _call_kwargs["filter_"] == {"doc_type": "pdf"}
