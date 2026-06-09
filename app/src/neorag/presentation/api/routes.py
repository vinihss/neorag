import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File, Form
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


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    document_id: str = ""


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


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    request: Request = None,
):
    container = _get_container(request)
    embedder = container.embedder()
    vector_store = container.vector_store()
    collection = container.config.qdrant.collection_name

    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader = container.loader_for(str(suffix.lower().lstrip(".")))
        docs = await loader.load(tmp_path)

        chunker = container.chunker()
        chunks = chunker.chunk(docs)

        texts = [c.content for c in chunks]
        vectors = await embedder.embed(texts)

        points = [
            {
                "id": c.id,
                "vector": v,
                "text": c.content,
                "source": c.source or file.filename,
                "page": c.page,
                "session_id": session_id,
                "doc_type": c.metadata.get("doc_type", ""),
            }
            for c, v in zip(chunks, vectors)
        ]

        if not await vector_store.collection_exists(collection):
            await vector_store.create_collection(
                collection, vector_size=embedder.dimensions
            )

        await vector_store.add(collection, points)

        return UploadResponse(
            filename=file.filename or "unknown",
            chunks=len(chunks),
            document_id=docs[0].id if docs else "",
        )
    finally:
        os.unlink(tmp_path)
