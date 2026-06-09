# AGENTS.md — Guia para Agentes de IA

## Projeto: NeoRAG

RAG Profissional Multilíngue. Hexagonal Architecture, Ports & Adapters, Python 3.12+, uv.

## Stack

| Camada | Tecnologia | Propósito |
|---|---|---|
| Web framework | FastAPI | API REST assíncrona |
| UI | Streamlit | Interface de chat |
| Vector DB | Qdrant | Armazenamento e busca de embeddings |
| LLM | Ollama + llama3.2:8b | Geração local gratuita |
| Embedding | BAAI/bge-m3 (sentence-transformers) | Multilíngue, 1024 dims |
| Orquestração | LangChain | Chunking, chains |
| Package | uv | Python package manager |

## Git Workflow

```
main ← dev ← phase/{numero}-{nome}
```

- **main**: produção (commits estáveis)
- **dev**: integração (merges das fases aprovadas)
- **phase/N-nome**: branch de trabalho, após testes mergeia em dev

```bash
git checkout dev
git checkout -b phase/N-nome
# ... implementar ...
# testar:
uv run pytest tests/ -v
uv run ruff check src/ tests/
# merge:
git checkout dev
git merge phase/N-nome --no-ff -m "feat: merge Phase N - descrição"
```

## Commits

Conventional Commits:

```
feat: nova funcionalidade
fix: correção de bug
docs: documentação
test: testes
refactor: refatoração
chore: tarefa técnica
```

## Arquitetura

### Camadas

```
Presentation (FastAPI/Streamlit)
    → Application (use cases/services)
        → Domain (entities + ports/Protocols)
            ← Infrastructure (implementações)
```

### Padrões

- **Ports & Adapters**: `domain/ports.py` define Protocols. `infrastructure/` implementa.
- **Strategy**: Retrieval (dense, sparse, hybrid), Chunkers, Loaders.
- **Factory**: `registry` dict + `register` decorator + `create` method.
- **Repository**: SessionRepository abstrai persistência (SQLite/Redis).
- **DI Container**: `presentation/container.py` — único lugar com imports concretos.

### Substituir Componente

1. Criar classe que implementa o Protocol
2. Registrar na Factory com `@register("nome")`
3. Configurar via `.env` — sem mudar código application/presentation

## Domain Ports (Protocols)

```python
Embedder       — embed(texts), embed_query(text), dimensions
VectorStore    — add(), search(), delete(), collection_exists(), create_collection()
DocumentLoader — load(source)
Chunker        — chunk(documents, chunk_size, overlap)
Reranker       — rerank(query, candidates, top_k)
LLM            — generate(prompt), generate_stream(prompt)
SessionRepo    — get(), save(), add_message(), get_history(), list_sessions()
```

## Para Implementar

### Fase 2 — Ingestion Pipeline
- `infrastructure/ingestion/loaders/` — PDF, HTML, MD, DOCX
- `infrastructure/ingestion/chunkers/` — Recursivo (LangChain)
- `infrastructure/ingestion/embedders/` — bge-m3
- `infrastructure/retrieval/vector_stores/` — Qdrant

### Fase 3 — Retrieval
- `infrastructure/retrieval/strategies/` — Dense, Sparse, Hybrid
- `infrastructure/retrieval/rerankers/` — Cross-Encoder

### Fase 4 — Geração
- `infrastructure/generation/llms/` — Ollama
- `application/generation.py`
- Prompt templates, session manager, API `/ask`

### Fase 5 — UI
- `ui/` — Streamlit app com upload e chat

## Comandos Frequentes

```bash
cd app
uv sync --all-extras        # instalar deps
uv pip install -e .         # modo editable
uv run pytest tests/ -v     # rodar testes
uv run pytest tests/unit/   # só unitários
uv run ruff check src/ tests/  # lint
uv run ruff format src/ tests/ # formatar
docker compose up --build   # subir tudo
```

## Config (.env)

Variáveis com prefixo da seção + `__`:

```
EMBEDDER__TYPE=bge-m3
QDRANT__HOST=qdrant
LLM__MODEL=llama3.2:8b
RETRIEVAL__ALPHA=0.5
```

## Estrutura dos Testes

```
tests/
├── unit/domain/        # Entidades + Ports (mocks)
├── unit/application/   # Serviços mockados
├── integration/        # Componentes reais
└── e2e/               # Pipeline completo
```

## Regras

1. Domain NUNCA importa infra ou framework externo
2. Infrastructure SÓ implementa interfaces do domain
3. Application SÓ depende de domain
4. Container é o ÚNICO lugar que sabe das implementações concretas
5. Todo componente substituível tem Protocol + Factory + config no .env
6. README.md é atualizado a cada fase concluída
