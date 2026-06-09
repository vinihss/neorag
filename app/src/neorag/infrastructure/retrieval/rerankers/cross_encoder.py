from sentence_transformers import CrossEncoder as SECrossEncoder

from neorag.domain.value_objects import ScoredChunk
from neorag.infrastructure.retrieval.rerankers.base import Reranker
from neorag.infrastructure.retrieval.rerankers.factory import RerankerFactory


@RerankerFactory.register("cross-encoder")
class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        self._model = SECrossEncoder(model_name, device=device)

    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for idx, score in indexed[:top_k]:
            chunk = candidates[idx]
            chunk.score = float(score)
            results.append(chunk)
        return results
