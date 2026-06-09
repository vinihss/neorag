import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)
    answer: str = ""


@dataclass
class EvalDataset:
    samples: list[EvalSample] = field(default_factory=list)

    def to_dicts(self) -> list[dict]:
        return [
            {
                "question": s.question,
                "ground_truth": s.ground_truth,
                "contexts": s.contexts,
                "answer": s.answer,
            }
            for s in self.samples
        ]

    @classmethod
    def from_dicts(cls, data: list[dict]) -> "EvalDataset":
        return cls(
            samples=[
                EvalSample(
                    question=d["question"],
                    ground_truth=d.get("ground_truth", ""),
                    contexts=d.get("contexts", []),
                    answer=d.get("answer", ""),
                )
                for d in data
            ]
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dicts(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "EvalDataset":
        with open(path) as f:
            return cls.from_dicts(json.load(f))


SAMPLE_DATASET = EvalDataset(
    samples=[
        EvalSample(
            question="What is RAG?",
            ground_truth=(
                "Retrieval-Augmented Generation (RAG) is a technique that "
                "combines retrieval of relevant documents with text generation "
                "to produce more accurate and context-aware responses."
            ),
        ),
        EvalSample(
            question="How does hybrid search work?",
            ground_truth=(
                "Hybrid search combines dense vector embeddings with sparse "
                "BM25 scoring, blending results using an alpha parameter to "
                "leverage both semantic and keyword-based retrieval."
            ),
        ),
    ]
)
