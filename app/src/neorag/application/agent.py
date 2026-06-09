import json
import re

from neorag.application.tools import ToolRegistry
from neorag.domain.entities import Answer, ScoredChunk
from neorag.domain.ports import LLM

AGENT_SYSTEM_PROMPT = """You are an intelligent RAG agent. You have access to the following tools:

{tools_description}

To use a tool, respond with EXACTLY this format (and nothing else):
TOOL: search_documents
ARGS: {{"query": "your search query", "top_k": 5}}

After receiving tool results, provide your final answer to the user.
If you don't need a tool, just answer directly.

Always cite your sources when using information from documents."""


class Agent:
    def __init__(self, llm: LLM, tools: ToolRegistry, max_steps: int = 3) -> None:
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps

    async def run(self, question: str, session_id: str = "default") -> Answer:
        tools_desc = "\n".join(
            f"- {t['name']}: {t['description']}"
            for t in self._tools.list_tools()
        )
        system = AGENT_SYSTEM_PROMPT.format(tools_description=tools_desc)

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": question})

        sources: list[ScoredChunk] = []

        for step in range(self._max_steps):
            prompt = self._build_prompt(messages)
            response = await self._llm.generate(prompt)

            if response.strip().startswith("TOOL:"):
                tool_result = await self._execute_tool_call(response)
                if tool_result.get("sources"):
                    sources.extend(tool_result["sources"])
                messages.append({"role": "assistant", "content": response})
                messages.append(
                    {"role": "system", "content": f"Tool result: {tool_result['text']}"}
                )
            else:
                return Answer(
                    text=response,
                    sources=sources,
                    session_id=session_id,
                    confidence=0.0,
                )

        return Answer(
            text="I couldn't find enough information to answer your question.",
            sources=sources,
            session_id=session_id,
            confidence=0.0,
        )

    def _build_prompt(self, messages: list[dict]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m["role"].capitalize()
            parts.append(f"{role}: {m['content']}")
        return "\n\n".join(parts) + "\n\nAssistant:"

    async def _execute_tool_call(self, response: str) -> dict:
        name_match = re.search(r"TOOL:\s*(\w+)", response)
        args_match = re.search(r"ARGS:\s*(\{.*\})", response, re.DOTALL)

        if not name_match:
            return {"text": "Invalid tool call format.", "sources": []}

        name = name_match.group(1)
        args: dict = {}
        if args_match:
            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                args = {}

        result_text = await self._tools.execute(name, **args)
        return {"text": result_text, "sources": []}
