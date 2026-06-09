import pytest
from unittest.mock import AsyncMock, MagicMock

from neorag.application.generation import GenerationService
from neorag.config import EvaluationConfig
from neorag.domain.entities import Answer
from neorag.domain.value_objects import ScoredChunk
from neorag.evaluation.dataset import EvalDataset, EvalSample
from neorag.evaluation.pipeline import EvaluationPipeline


@pytest.fixture
def mock_generation():
    m = MagicMock(spec=GenerationService)
    m.answer = AsyncMock(
        return_value=Answer(
            text="RAG combines retrieval and generation.",
            sources=[
                ScoredChunk(
                    chunk_id="c1",
                    score=0.9,
                    text="RAG is retrieval augmented generation.",
                    source="doc1",
                )
            ],
            session_id="eval",
            confidence=0.9,
        )
    )
    return m


class TestEvaluationPipeline:
    @pytest.mark.asyncio
    async def test_run_returns_metrics_structure(self, mock_generation):
        config = EvaluationConfig()
        pipeline = EvaluationPipeline(generation=mock_generation, config=config)

        dataset = EvalDataset(
            samples=[EvalSample(question="What is RAG?", ground_truth="RAG is...")]
        )

        result = await pipeline.run(dataset)

        assert "metrics" in result
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["question"] == "What is RAG?"

    @pytest.mark.asyncio
    async def test_run_multiple_samples(self, mock_generation):
        pipeline = EvaluationPipeline(generation=mock_generation)

        dataset = EvalDataset(
            samples=[
                EvalSample(question="Q1", ground_truth="A1"),
                EvalSample(question="Q2", ground_truth="A2"),
            ]
        )

        result = await pipeline.run(dataset)
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_run_empty_dataset(self, mock_generation):
        pipeline = EvaluationPipeline(generation=mock_generation)
        dataset = EvalDataset(samples=[])
        result = await pipeline.run(dataset)

        assert result["metrics"] == {}
        assert result["results"] == []
