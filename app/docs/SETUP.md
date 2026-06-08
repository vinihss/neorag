# Setup — NeoRAG

## Pré-requisitos

- Docker + Docker Compose (v2.22+)
- Git
- 8GB+ RAM (recomendado 16GB para rodar LLM local)
- 10GB+ espaço em disco (modelos ~4GB)

## Quick Start

```bash
# 1. Clone
git clone <repo> && cd neorag

# 2. Configure variáveis (opcional — defaults funcionam)
cp .env.example .env

# 3. Inicie tudo
docker compose up --build

# 4. Acesse
#    API:    http://localhost:8000/docs
#    UI:     http://localhost:8501
#    Qdrant: http://localhost:6333/dashboard

# 5. (Primeira vez) Baixe o modelo LLM
docker compose exec ollama ollama pull llama3.2:8b
```

## Serviços

| Serviço | Porta | Descrição |
|---|---|---|
| `app` | 8000 | API FastAPI + OpenAPI docs |
| `ui` | 8501 | Interface Streamlit |
| `qdrant` | 6333 / 6334 | Vector DB (REST / GRPC) |
| `ollama` | 11434 | LLM server |

## Variáveis de Ambiente (.env)

```env
# --- Embedder ---
EMBEDDER__TYPE=bge-m3
EMBEDDER__MODEL_NAME=BAAI/bge-m3
EMBEDDER__DEVICE=cpu

# --- Qdrant ---
QDRANT__HOST=qdrant
QDRANT__PORT=6333
QDRANT__COLLECTION_NAME=documents

# --- LLM ---
LLM__TYPE=ollama
LLM__BASE_URL=http://ollama:11434
LLM__MODEL=llama3.2:8b
LLM__TEMPERATURE=0.1
LLM__MAX_TOKENS=1024

# --- Retrieval ---
RETRIEVAL__ALPHA=0.5
RETRIEVAL__TOP_K_RETRIEVE=20
RETRIEVAL__TOP_K_RERANK=5
RETRIEVAL__MIN_SCORE=0.6

# --- Session ---
SESSION__STORAGE=sqlite
SESSION__DB_PATH=data/sessions.db
```

## Comandos Úteis

```bash
# Logs
docker compose logs -f app      # API
docker compose logs -f ui       # UI

# Executar comandos no container
docker compose exec app python -m neorag.main --help

# Baixar modelo específico
docker compose exec ollama ollama pull qwen2.5:7b

# Rebuild
docker compose up --build -d

# Reset (cuidado: apaga dados)
docker compose down -v
```

## Testes

```bash
# Unitários
docker compose exec app python -m pytest tests/unit -v

# Integração
docker compose exec app python -m pytest tests/integration -v

# E2E (requer todos os serviços rodando)
docker compose exec app python -m pytest tests/e2e -v

# Todos
docker compose exec app python -m pytest tests/ -v --cov=src/neorag
```

## Estrutura de Dados

```bash
data/
├── documents/       # Documentos de referência (montados no container)
├── uploads/         # Uploads dos usuários
├── sessions.db      # SQLite com histórico de conversas
├── qdrant/          # Dados persistentes do Qdrant
└── ollama/          # Modelos baixados pelo Ollama
```
