import pytest

from neorag.domain.value_objects import ScoredChunk
from neorag.infrastructure.retrieval.rerankers.factory import RerankerFactory


class TestRerankerFactory:
    def test_registered_rerankers(self):
        assert "cross-encoder" in RerankerFactory._registry

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown reranker"):
            RerankerFactory.create("unknown")


@pytest.mark.skip(reason="Requires cross-encoder model download")
class TestCrossEncoderReranker:
    @pytest.mark.asyncio
    async def test_rerank_returns_sorted_results(self):
        reranker = RerankerFactory.create("cross-encoder")
        candidates = [
            ScoredChunk(chunk_id="c1", score=0.0, text="cat sat on mat"),
            ScoredChunk(chunk_id="c2", score=0.0, text="dogs love parks"),
            ScoredChunk(chunk_id="c3", score=0.0, text="cats are pets"),
        ]

        results = await reranker.rerank("cat", candidates, top_k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert all(r.chunk_id in {"c1", "c2", "c3"} for r in results)

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self):
        reranker = RerankerFactory.create("cross-encoder")
        results = await reranker.rerank("query", [], top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_single_candidate(self):
        reranker = RerankerFactory.create("cross-encoder")
        candidates = [
            ScoredChunk(chunk_id="c1", score=0.0, text="single document"),
        ]

        results = await reranker.rerank("test", candidates, top_k=5)
        assert len(results) == 1
        assert results[0].chunk_id == "c1"
