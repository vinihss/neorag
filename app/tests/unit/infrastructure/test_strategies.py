import pytest
from unittest.mock import AsyncMock, MagicMock

from neorag.domain.ports import Embedder, VectorStore
from neorag.domain.value_objects import ScoredChunk
from neorag.infrastructure.retrieval.strategies.factory import RetrievalStrategyFactory
from neorag.infrastructure.retrieval.strategies.dense import DenseRetrievalStrategy
from neorag.infrastructure.retrieval.strategies.sparse import SparseRetrievalStrategy
from neorag.infrastructure.retrieval.strategies.hybrid import HybridRetrievalStrategy


class TestRetrievalStrategyFactory:
    def test_registered_strategies(self):
        assert "dense" in RetrievalStrategyFactory._registry
        assert "sparse" in RetrievalStrategyFactory._registry
        assert "hybrid" in RetrievalStrategyFactory._registry

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            RetrievalStrategyFactory.create("unknown")


class TestDenseRetrievalStrategy:
    @pytest.mark.asyncio
    async def test_retrieve_returns_scored_chunks(self):
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_store = MagicMock(spec=VectorStore)
        expected = [
            ScoredChunk(chunk_id="c1", score=0.9, text="doc1"),
            ScoredChunk(chunk_id="c2", score=0.8, text="doc2"),
        ]
        mock_store.search = AsyncMock(return_value=expected)

        strategy = DenseRetrievalStrategy(
            embedder=mock_embedder,
            vector_store=mock_store,
            collection_name="test",
        )

        results = await strategy.retrieve("query", k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"
        mock_embedder.embed_query.assert_called_once_with("query")
        mock_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_passes_filter(self):
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_store = MagicMock(spec=VectorStore)
        mock_store.search = AsyncMock(return_value=[])

        strategy = DenseRetrievalStrategy(
            embedder=mock_embedder,
            vector_store=mock_store,
            collection_name="test",
        )

        await strategy.retrieve("query", k=5, filter_={"user_id": "u1"})

        _call_kwargs = mock_store.search.call_args.kwargs
        assert _call_kwargs["filter_"] == {"user_id": "u1"}

    @pytest.mark.asyncio
    async def test_retrieve_empty_result(self):
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        mock_store = MagicMock(spec=VectorStore)
        mock_store.search = AsyncMock(return_value=[])

        strategy = DenseRetrievalStrategy(
            embedder=mock_embedder,
            vector_store=mock_store,
            collection_name="test",
        )

        results = await strategy.retrieve("query", k=5)
        assert results == []


class TestSparseRetrievalStrategy:
    def test_bm25_index_building(self):
        strategy = SparseRetrievalStrategy(
            texts=["hello world", "foo bar baz", "hello hello"],
            ids=["d1", "d2", "d3"],
        )
        assert strategy._bm25 is not None

    @pytest.mark.asyncio
    async def test_retrieve_returns_expected(self):
        strategy = SparseRetrievalStrategy(
            texts=[
                "the cat sat on the mat",
                "dogs love to run in the park",
                "cats and dogs are pets",
            ],
            ids=["d1", "d2", "d3"],
        )

        results = await strategy.retrieve("cat", k=3)
        assert len(results) == 3
        assert results[0].chunk_id == "d1"
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self):
        strategy = SparseRetrievalStrategy(
            texts=["a b c", "d e f", "g h i"],
            ids=["d1", "d2", "d3"],
        )

        results = await strategy.retrieve("a", k=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_empty_corpus(self):
        strategy = SparseRetrievalStrategy()
        results = await strategy.retrieve("anything", k=5)
        assert results == []

    def test_index_method(self):
        strategy = SparseRetrievalStrategy()
        assert strategy._bm25 is None

        strategy.index(
            texts=["new text one", "new text two"],
            ids=["n1", "n2"],
        )
        assert strategy._bm25 is not None

    def test_add_method(self):
        strategy = SparseRetrievalStrategy(
            texts=["initial"], ids=["i1"]
        )
        assert len(strategy._texts) == 1

        strategy.add("added text", "a1")
        assert len(strategy._texts) == 2
        assert strategy._texts[1] == "added text"


class TestHybridRetrievalStrategy:
    @pytest.mark.asyncio
    async def test_combines_dense_and_sparse(self):
        mock_dense = AsyncMock(spec=DenseRetrievalStrategy)
        mock_dense.retrieve = AsyncMock(
            return_value=[
                ScoredChunk(chunk_id="c1", score=0.9, text="dense doc1"),
                ScoredChunk(chunk_id="c2", score=0.8, text="dense doc2"),
            ]
        )

        mock_sparse = AsyncMock(spec=SparseRetrievalStrategy)
        mock_sparse.retrieve = AsyncMock(
            return_value=[
                ScoredChunk(chunk_id="c1", score=10.0, text="sparse doc1"),
                ScoredChunk(chunk_id="c3", score=5.0, text="sparse doc3"),
            ]
        )

        strategy = HybridRetrievalStrategy(
            dense=mock_dense, sparse=mock_sparse, alpha=0.5
        )

        results = await strategy.retrieve("query", k=3)

        assert len(results) == 3
        chunk_ids = [r.chunk_id for r in results]
        assert "c1" in chunk_ids
        assert "c2" in chunk_ids
        assert "c3" in chunk_ids

    @pytest.mark.asyncio
    async def test_alpha_controls_weighting(self):
        mock_dense = AsyncMock(spec=DenseRetrievalStrategy)
        mock_dense.retrieve = AsyncMock(
            return_value=[
                ScoredChunk(chunk_id="c1", score=0.9, text="dense"),
            ]
        )

        mock_sparse = AsyncMock(spec=SparseRetrievalStrategy)
        mock_sparse.retrieve = AsyncMock(
            return_value=[
                ScoredChunk(chunk_id="c2", score=10.0, text="sparse"),
            ]
        )

        strategy_pure_dense = HybridRetrievalStrategy(
            dense=mock_dense, sparse=mock_sparse, alpha=1.0
        )
        results = await strategy_pure_dense.retrieve("query", k=5)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_sparse_results(self):
        mock_dense = AsyncMock(spec=DenseRetrievalStrategy)
        mock_dense.retrieve = AsyncMock(
            return_value=[
                ScoredChunk(chunk_id="c1", score=0.9, text="only dense"),
            ]
        )

        mock_sparse = AsyncMock(spec=SparseRetrievalStrategy)
        mock_sparse.retrieve = AsyncMock(return_value=[])

        strategy = HybridRetrievalStrategy(
            dense=mock_dense, sparse=mock_sparse, alpha=0.5
        )
        results = await strategy.retrieve("query", k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
