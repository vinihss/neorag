# NeoRAG — RAG Profissional Multilíngue

Sistema Retrieval-Augmented Generation multilíngue, open-source, modular e extensível.

## Stack

- **Embedding**: BAAI/bge-m3 (multilíngue, dense+sparse)
- **Vector DB**: Qdrant
- **LLM**: Ollama + Llama 3.2 (local, gratuito)
- **Orquestração**: LangChain + FastAPI
- **UI**: Streamlit
- **Infra**: Docker Compose

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
# UI: http://localhost:8501
# API: http://localhost:8000/docs
docker compose exec ollama ollama pull llama3.2:8b
```

## Entregas

| Fase | Descrição | Status |
|---|---|---|
| 1 | Infraestrutura + Domain + Config | △ |
| 2 | Ingestion (loaders, chunkers, embedders, Qdrant) | △ |
| 3 | Retrieval (hybrid search, reranker) | △ |
| 4 | Geração (LLM, prompts, API, sessões) | △ |
| 5 | UI Streamlit | △ |
| 6 | Avaliação (RAGAS, monitoramento) | △ |
| 7 | Multi-tenancy + Agentic | △ |

## Testes

3 níveis: **unit** (mock), **integration** (real), **e2e** (pipeline).

```bash
pytest tests/unit/        # ~60 testes
pytest tests/integration/ # ~52 testes
pytest tests/e2e/         # ~16 testes
```

## Documentos

| Arquivo | Conteúdo |
|---|---|
| `docs/ARCHITECTURE.md` | Diagramas, fluxos, responsabilidades |
| `docs/DECISIONS.md` | Decisões técnicas com alternativas |
| `docs/SETUP.md` | Setup e comandos |
| `docs/TEST_PLAN.md` | 3 níveis de teste por feature (128 testes) |
