from dataclasses import dataclass, field


@dataclass
class ScoredChunk:
    chunk_id: str = ""
    score: float = 0.0
    text: str = ""
    source: str = ""
    page: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Source:
    title: str = ""
    url: str = ""
    page: int | None = None
    snippet: str = ""
