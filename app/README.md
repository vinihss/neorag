# NeoRAG — RAG Profissional Multilíngue

Sistema Retrieval-Augmented Generation multilíngue, open-source, modular e extensível.

## Stack

- **Embedding**: BAAI/bge-m3 (multilíngue, dense+sparse)
- **Vector DB**: Qdrant
- **LLM**: Ollama + Llama 3.2 (local, gratuito)
- **Orquestração**: LangChain + FastAPI
- **UI**: Streamlit
- **Infra**: Docker Compose
- **Package Manager**: uv (Python 3.12+)

## Arquitetura

```
Camadas: Presentation → Application → Domain ← Infrastructure
Padrões: Ports & Adapters, Strategy, Factory, Repository, DI Container
```

[Documentação completa →](docs/ARCHITECTURE.md)

## Quick Start

```bash
cp .env.example .env
docker compose up --build
# API: http://localhost:8000/docs
docker compose exec ollama ollama pull llama3.2:8b
```

## Entregas

| Fase | Descrição | Status |
|---|---|---|
| 1 | Infraestrutura + Domain + Config | ✅ |
| 2 | Ingestion (loaders, chunkers, embedders, Qdrant) | △ |
| 3 | Retrieval (hybrid search, reranker) | △ |
| 4 | Geração (LLM, prompts, API, sessões) | △ |
| 5 | UI Streamlit | △ |
| 6 | Avaliação (RAGAS, monitoramento) | △ |
| 7 | Multi-tenancy + Agentic | △ |

## Fase 1 — Infraestrutura Base ✓

Commit: `153fb1d` — Branch: `phase/1-infra` (mergeada em `dev`)

### Entregues

| Componente | Descrição |
|---|---|
| **Git** | Configurado: Vinicius Heleno <vinihss@gmail.com>, remote: github.com/vinihss/neorag |
| **Branches** | `main`, `dev`, `phase/1-infra` |
| **Docker Compose** | 3 serviços: `app` (FastAPI :8000), `qdrant` (:6333), `ollama` (:11434) |
| **Dockerfile** | Multi-stage com uv, .venh isolado, imagem final slim |
| **pyproject.toml** | uv, deps (fastapi, pydantic, uvicorn) + dev (pytest, ruff, httpx) |
| **Config** | Pydantic Settings por contexto: embedder, qdrant, llm, retrieval, session |
| **Domain Entities** | Document, Chunk, Query, Answer, Message, Session |
| **Value Objects** | ScoredChunk, Source |
| **Domain Ports** | 7 Protocols: Embedder, VectorStore, LLM, Chunker, Loader, Reranker, SessionRepo |
| **DI Container** | Composition root em `presentation/container.py` |
| **API** | FastAPI com `/health` endpoint |
| **Docs** | ARCHITECTURE.md, DECISIONS.md, SETUP.md, TEST_PLAN.md |

### Testes: 45/45 passando

```bash
pytest tests/unit/domain/           # 28 testes (entities + ports)
pytest tests/integration/test_config.py  # 17 testes (settings)
```

### Estrutura de Diretórios

```
neorag/
├── docker-compose.yml
├── app/
│   ├── Dockerfile
│   ├── pyproject.toml
│   └── src/neorag/
│       ├── main.py              ← FastAPI entry point
│       ├── config.py            ← Pydantic Settings
│       ├── domain/              ← Pure Python, sem dependências
│       │   ├── entities.py
│       │   ├── value_objects.py
│       │   └── ports.py         ← Protocols (interfaces)
│       ├── application/         ← Use cases (esqueletos)
│       ├── infrastructure/      ← Implementações (esqueletos)
│       └── presentation/
│           └── container.py     ← DI wiring
```

## Testes

3 níveis: **unit** (mock), **integration** (real), **e2e** (pipeline).

```bash
cd app
uv run pytest tests/unit/        # ~60 testes
uv run pytest tests/integration/ # ~52 testes
uv run pytest tests/e2e/         # ~16 testes
```

## Documentos

| Arquivo | Conteúdo |
|---|---|
| `docs/ARCHITECTURE.md` | Diagramas, fluxos, responsabilidades |
| `docs/DECISIONS.md` | Decisões técnicas com alternativas |
| `docs/SETUP.md` | Setup e comandos |
| `docs/TEST_PLAN.md` | 3 níveis de teste por feature (128 testes) |
