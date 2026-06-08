from typing import Protocol, runtime_checkable, AsyncIterator, Any

from neorag.domain.entities import Document, Chunk, Session, Message
from neorag.domain.value_objects import ScoredChunk


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...

    @property
    def dimensions(self) -> int: ...


@runtime_checkable
class VectorStore(Protocol):
    async def add(
        self, collection: str, points: list[dict[str, Any]]
    ) -> None: ...
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        k: int = 10,
        filter_: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]: ...
    async def delete(
        self, collection: str, point_ids: list[str]
    ) -> None: ...
    async def collection_exists(self, collection: str) -> bool: ...
    async def create_collection(
        self, collection: str, vector_size: int
    ) -> None: ...


@runtime_checkable
class DocumentLoader(Protocol):
    async def load(self, source: str) -> list[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(
        self,
        documents: list[Document],
        chunk_size: int = 512,
        overlap: float = 0.15,
    ) -> list[Chunk]: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int = 5,
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class LLM(Protocol):
    async def generate(self, prompt: str, **kwargs: Any) -> str: ...
    async def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class SessionRepository(Protocol):
    async def get(self, session_id: str) -> Session | None: ...
    async def save(self, session: Session) -> None: ...
    async def add_message(
        self, session_id: str, message: Message
    ) -> None: ...
    async def get_history(
        self, session_id: str, limit: int = 10
    ) -> list[Message]: ...
    async def list_sessions(
        self, user_id: str | None = None
    ) -> list[Session]: ...
