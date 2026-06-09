from typing import Protocol, runtime_checkable, AsyncIterator, Any


@runtime_checkable
class LLM(Protocol):
    async def generate(self, prompt: str, **kwargs: Any) -> str: ...
    async def generate_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[str]: ...
