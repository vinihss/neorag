from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from typing import Any

from neorag.domain.value_objects import ScoredChunk, Source


@dataclass
class Document:
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    doc_type: str = ""

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value


@dataclass
class Chunk:
    id: str = field(default_factory=lambda: str(uuid4()))
    document_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    page: int | None = None
    source: str = ""


@dataclass
class Query:
    text: str = ""
    top_k: int = 5
    filter: dict[str, Any] = field(default_factory=dict)
    user_id: str | None = None


@dataclass
class Answer:
    text: str = ""
    sources: list[ScoredChunk] = field(default_factory=list)
    session_id: str = ""
    confidence: float = 0.0


@dataclass
class Message:
    role: str = ""
    content: str = ""
    sources: list[Source] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid4()))
    user_id: str = ""
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def last_messages(self, limit: int = 10) -> list[Message]:
        return self.messages[-limit:]
