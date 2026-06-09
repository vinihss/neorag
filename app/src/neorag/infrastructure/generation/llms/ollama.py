from typing import Any, AsyncIterator

import httpx

from neorag.infrastructure.generation.llms.base import LLM
from neorag.infrastructure.generation.llms.factory import LLMFactory


@LLMFactory.register("ollama")
class OllamaLLM(LLM):
    def __init__(
        self,
        base_url: str = "http://ollama:11434",
        model: str = "llama3.2:8b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", self._temperature),
                    "max_tokens": kwargs.get("max_tokens", self._max_tokens),
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")

    async def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", self._temperature),
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        import json
                        try:
                            data = json.loads(line)
                            token = data.get("response", "")
                            if token:
                                yield token
                            if data.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
