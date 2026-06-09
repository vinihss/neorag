import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from neorag.infrastructure.generation.llms.factory import LLMFactory
from neorag.infrastructure.generation.llms.ollama import OllamaLLM


class TestLLMFactory:
    def test_registered_llms(self):
        assert "ollama" in LLMFactory._registry

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown LLM"):
            LLMFactory.create("unknown")


class TestOllamaLLM:
    @pytest.mark.asyncio
    async def test_generate_returns_text(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"response": "Hello from Ollama"})

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        llm = OllamaLLM(base_url="http://localhost:11434", model="test-model")

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await llm.generate("test prompt")

        assert result == "Hello from Ollama"

    @pytest.mark.asyncio
    async def test_generate_empty_response(self):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = MagicMock(return_value={"response": ""})

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)

        llm = OllamaLLM()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await llm.generate("test")

        assert result == ""
