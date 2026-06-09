from neorag.application.generation import GenerationService
from neorag.config import EvaluationConfig
from neorag.domain.entities import Query
from neorag.evaluation.dataset import EvalDataset


class EvaluationPipeline:
    def __init__(
        self,
        generation: GenerationService,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._generation = generation
        self._config = config or EvaluationConfig()

    async def run(self, dataset: EvalDataset) -> dict:
        results: list[dict] = []

        for sample in dataset.samples:
            query = Query(text=sample.question, top_k=5)
            answer = await self._generation.answer(query)
            contexts = [s.text for s in answer.sources]
            results.append(
                {
                    "question": sample.question,
                    "answer": answer.text,
                    "contexts": contexts,
                    "ground_truth": sample.ground_truth,
                }
            )

        if not results:
            return {"metrics": {}, "results": []}

        scores = await self._compute_metrics(results)
        return {"metrics": scores, "results": results}

    async def _compute_metrics(self, results: list[dict]) -> dict:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            metric_map = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }

            selected = [metric_map[m] for m in self._config.metrics if m in metric_map]
            if not selected:
                return {}

            ds = Dataset.from_list(results)
            result = evaluate(ds, metrics=selected)
            return {m: float(result[m]) for m in self._config.metrics if m in result}
        except ImportError:
            return {"error": "ragas or datasets not installed"}
