from typing import Any, Protocol

from neorag.domain.entities import Query
from neorag.application.retrieval import RetrievalService


class Tool(Protocol):
    name: str
    description: str

    async def execute(self, **kwargs: Any) -> str: ...


class SearchDocumentsTool:
    name = "search_documents"
    description = "Search the document database for relevant information. Use this when you need to find specific facts, data, or context from the user's documents."

    def __init__(self, retrieval: RetrievalService) -> None:
        self._retrieval = retrieval

    async def execute(self, query: str, top_k: int = 5, **kwargs: Any) -> str:
        q = Query(text=query, top_k=top_k)
        results = await self._retrieval.retrieve(q)
        if not results:
            return "No relevant documents found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            source = r.source or "unknown"
            lines.append(f"[{i}] (score: {r.score:.3f}, source: {source})\n{r.text}")
        return "\n\n".join(lines)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    async def execute(self, name: str, **kwargs: Any) -> str:
        tool = self.get(name)
        if not tool:
            return f"Unknown tool: {name}"
        return await tool.execute(**kwargs)
