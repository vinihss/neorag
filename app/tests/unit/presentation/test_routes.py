import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from neorag.main import app
from neorag.presentation.container import Container
from neorag.application.generation import GenerationService


@pytest.fixture
def mock_gen_service():
    m = MagicMock(spec=GenerationService)
    m.answer = AsyncMock(
        return_value=MagicMock(
            text="Answer text",
            session_id="s1",
            sources=[],
        )
    )
    return m


@pytest.fixture
def app_with_mock(mock_gen_service):
    container = MagicMock(spec=Container)
    container.generation_service = MagicMock(return_value=mock_gen_service)
    app.state.container = container
    yield app


@pytest.mark.asyncio
class TestAskRoute:
    async def test_ask_returns_answer(self, app_with_mock, mock_gen_service):
        transport = ASGITransport(app=app_with_mock)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/ask",
                json={"question": "What is RAG?", "session_id": "test123"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Answer text"
        assert data["session_id"] == "s1"

    async def test_ask_missing_question(self, app_with_mock):
        transport = ASGITransport(app=app_with_mock)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/ask",
                json={"session_id": "test"},
            )

        assert response.status_code == 422

    async def test_health_endpoint(self, app_with_mock):
        transport = ASGITransport(app=app_with_mock)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
