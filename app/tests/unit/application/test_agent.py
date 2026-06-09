import pytest
from unittest.mock import AsyncMock, MagicMock

from neorag.application.agent import Agent
from neorag.application.tools import ToolRegistry, SearchDocumentsTool
from neorag.application.retrieval import RetrievalService
from neorag.domain.ports import LLM


@pytest.fixture
def mock_llm():
    m = MagicMock(spec=LLM)
    return m


@pytest.fixture
def mock_retrieval():
    m = MagicMock(spec=RetrievalService)
    m.retrieve = AsyncMock(return_value=[])
    return m


class TestToolRegistry:
    def test_register_and_list(self):
        registry = ToolRegistry()
        registry.register(SearchDocumentsTool(retrieval=MagicMock()))
        assert len(registry.list_tools()) == 1
        assert registry.list_tools()[0]["name"] == "search_documents"

    def test_get_nonexistent(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("unknown")
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_search_tool_no_results(self, mock_retrieval):
        mock_retrieval.retrieve = AsyncMock(return_value=[])
        tool = SearchDocumentsTool(retrieval=mock_retrieval)
        result = await tool.execute(query="test")
        assert "No relevant documents found" in result


class TestAgent:
    @pytest.mark.asyncio
    async def test_direct_answer(self, mock_llm):
        mock_llm.generate = AsyncMock(return_value="Direct answer without tools.")
        tools = ToolRegistry()
        agent = Agent(llm=mock_llm, tools=tools)

        answer = await agent.run("Hello")
        assert answer.text == "Direct answer without tools."

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self, mock_llm):
        mock_llm.generate = AsyncMock(
            return_value='TOOL: nonexistent\nARGS: {"query": "test"}'
        )
        tools = ToolRegistry()
        agent = Agent(llm=mock_llm, tools=tools, max_steps=1)

        answer = await agent.run("test")
        assert "couldn't find" in answer.text.lower()
