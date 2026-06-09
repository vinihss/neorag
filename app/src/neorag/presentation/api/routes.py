from fastapi import APIRouter, Request
from pydantic import BaseModel

from neorag.domain.entities import Query

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"
    user_id: str | None = None
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[dict] = []


def _get_container(request: Request):
    return request.app.state.container


@router.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@router.post("/ask", response_model=AskResponse)
async def ask(
    req: AskRequest,
    request: Request,
):
    container = _get_container(request)
    gen = container.generation_service()

    query = Query(
        text=req.question,
        top_k=req.top_k,
        user_id=req.user_id,
    )
    if req.session_id != "default":
        query.filter["session_id"] = req.session_id

    answer = await gen.answer(query)

    return AskResponse(
        answer=answer.text,
        session_id=answer.session_id,
        sources=[
            {
                "title": s.metadata.get("title", ""),
                "source": s.source,
                "page": s.page,
                "score": s.score,
                "text": s.text[:500],
            }
            for s in answer.sources
        ],
    )
