import textwrap

from neorag.application.prompts import CHAT_PROMPT, SYSTEM_PROMPT
from neorag.application.retrieval import RetrievalService
from neorag.domain.entities import Answer, Message, Query, Session
from neorag.domain.ports import LLM, SessionRepository
from neorag.domain.value_objects import ScoredChunk, Source

MAX_CONTEXT_CHARS = 8000


def _format_context(chunks: list[ScoredChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.source or "unknown"
        parts.append(f"[{i}] (source: {source})\n{chunk.text}")
    return "\n\n".join(parts)


def _chunks_to_sources(chunks: list[ScoredChunk]) -> list[Source]:
    seen: set[str] = set()
    sources: list[Source] = []
    for c in chunks:
        key = f"{c.source}:{c.page}" if c.page else c.source
        if key not in seen:
            seen.add(key)
            sources.append(
                Source(
                    title=c.metadata.get("title", ""),
                    url=c.source,
                    page=c.page,
                    snippet=textwrap.shorten(c.text, width=200, placeholder="..."),
                )
            )
    return sources


class GenerationService:
    def __init__(
        self,
        llm: LLM,
        retrieval: RetrievalService,
        session_repo: SessionRepository,
    ) -> None:
        self._llm = llm
        self._retrieval = retrieval
        self._session_repo = session_repo

    async def answer(self, query: Query) -> Answer:
        session = await self._session_repo.get(query.user_id or "default")
        if not session:
            session = Session(id=query.user_id or "default", user_id=query.user_id)
            await self._session_repo.save(session)

        chunks = await self._retrieval.retrieve(query)

        context = _format_context(chunks)
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

        history_text = "\n".join(
            f"{m.role}: {m.content}" for m in session.last_messages(6)
        )

        system = SYSTEM_PROMPT.format(context=context)
        prompt = CHAT_PROMPT.format(
            system=system,
            history=history_text,
            question=query.text,
        )

        response = await self._llm.generate(prompt)

        sources = _chunks_to_sources(chunks)
        avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

        user_msg = Message(role="user", content=query.text)
        assistant_msg = Message(
            role="assistant",
            content=response,
            sources=sources,
        )
        await self._session_repo.add_message(session.id, user_msg)
        await self._session_repo.add_message(session.id, assistant_msg)

        return Answer(
            text=response,
            sources=chunks,
            session_id=session.id,
            confidence=avg_score,
        )
