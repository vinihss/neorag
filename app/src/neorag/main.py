from fastapi import FastAPI
from neorag.config import Settings
from neorag.presentation.container import Container
from neorag.presentation.api.routes import router

config = Settings()
container = Container(config)

app = FastAPI(
    title="NeoRAG",
    description="Professional multilingual RAG system",
    version="0.1.0",
)

app.state.container = container
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
