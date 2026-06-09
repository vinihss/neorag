import pytest
from unittest.mock import AsyncMock, MagicMock

from neorag.application.generation import GenerationService
from neorag.domain.entities import Query, Session, Message
from neorag.domain.ports import LLM, SessionRepository
from neorag.domain.value_objects import ScoredChunk
from neorag.application.retrieval import RetrievalService


@pytest.fixture
def mock_llm():
    m = MagicMock(spec=LLM)
    m.generate = AsyncMock(return_value="This is the answer based on the context.")
    return m


@pytest.fixture
def mock_retrieval():
    m = MagicMock(spec=RetrievalService)
    m.retrieve = AsyncMock(
        return_value=[
            ScoredChunk(chunk_id="c1", score=0.9, text="Relevant content about RAG.", source="doc1.pdf"),
        ]
    )
    return m


@pytest.fixture
def mock_session_repo():
    m = MagicMock(spec=SessionRepository)
    m.get = AsyncMock(return_value=None)
    m.save = AsyncMock()
    m.add_message = AsyncMock()
    return m


class TestGenerationService:
    @pytest.mark.asyncio
    async def test_answer_returns_answer(self, mock_llm, mock_retrieval, mock_session_repo):
        service = GenerationService(
            llm=mock_llm,
            retrieval=mock_retrieval,
            session_repo=mock_session_repo,
        )

        query = Query(text="What is RAG?", top_k=3)
        answer = await service.answer(query)

        assert answer.text == "This is the answer based on the context."
        assert len(answer.sources) == 1
        assert answer.sources[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_answer_creates_session(self, mock_llm, mock_retrieval, mock_session_repo):
        mock_session_repo.get = AsyncMock(return_value=None)

        service = GenerationService(
            llm=mock_llm,
            retrieval=mock_retrieval,
            session_repo=mock_session_repo,
        )

        query = Query(text="Hello", user_id="new_user")
        await service.answer(query)

        mock_session_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_uses_existing_session(self, mock_llm, mock_retrieval, mock_session_repo):
        existing = Session(id="existing", user_id="u1")
        existing.add_message(Message(role="user", content="previous question"))
        existing.add_message(Message(role="assistant", content="previous answer"))
        mock_session_repo.get = AsyncMock(return_value=existing)

        service = GenerationService(
            llm=mock_llm,
            retrieval=mock_retrieval,
            session_repo=mock_session_repo,
        )

        query = Query(text="Follow up question", user_id="u1")
        await service.answer(query)

        mock_llm.generate.assert_called_once()
        prompt = mock_llm.generate.call_args[0][0]
        assert "previous question" in prompt

    @pytest.mark.asyncio
    async def test_answer_stores_messages(self, mock_llm, mock_retrieval, mock_session_repo):
        mock_session_repo.get = AsyncMock(return_value=Session(id="s1", user_id="u1"))

        service = GenerationService(
            llm=mock_llm,
            retrieval=mock_retrieval,
            session_repo=mock_session_repo,
        )

        query = Query(text="Hi", user_id="u1")
        await service.answer(query)

        assert mock_session_repo.add_message.call_count == 2
