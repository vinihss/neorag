# Arquitetura NeoRAG

## Diagrama de Componentes

```
┌──────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION                                │
│                                                                      │
│  ┌─────────────────────┐          ┌──────────────────────────┐       │
│  │    FastAPI (8000)     │◄────────►│   Streamlit UI (8501)    │       │
│  │  ▪ /api/ask           │  REST   │  ▪ Upload documentos     │       │
│  │  ▪ /api/documents     │         │  ▪ Chat Q&A              │       │
│  │  ▪ /api/sessions      │         │  ▪ Visualizar fontes     │       │
│  │  ▪ /health            │         │  ▪ Histórico por sessão  │       │
│  └──────────┬──────────┘          └──────────────────────────┘       │
│             │                                                        │
└─────────────│────────────────────────────────────────────────────────┘
              │
┌─────────────│────────────────────────────────────────────────────────┐
│             ▼                 APPLICATION                             │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │                    Use Case / Services                       │     │
│  │                                                              │     │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐  │     │
│  │  │ IngestionService  │  │ RetrievalService  │  │GenService │  │     │
│  │  │  ▪ load_document  │  │  ▪ ask           │  │  ▪ stream │  │     │
│  │  │  ▪ process_file   │  │  ▪ retrieve      │  │  ▪ format │  │     │
│  │  └────────┬─────────┘  └────────┬─────────┘  └─────┬─────┘  │     │
│  └─────────────────────────────────────────────────────────────┘     │
│              │                 │                │                     │
└──────────────│─────────────────│────────────────│─────────────────────┘
               │                 │                │
        ┌──────▼─────────────────▼────────────────▼──────┐
        │                   DOMAIN                         │
        │  Entities: Document, Chunk, Query, Answer,      │
        │            Session, Message                     │
        │  Ports (Protocols):                             │
        │    ▪ DocumentLoader  ▪ Chunker    ▪ Embedder    │
        │    ▪ VectorStore     ▪ Retriever  ▪ Reranker   │
        │    ▪ LLM             ▪ SessionRepo              │
        └──────────────────────┬──────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                       INFRASTRUCTURE                                 │
│                                                                     │
│  ┌─────────────────────┐  ┌───────────────────────────┐             │
│  │     INGESTION        │  │       RETRIEVAL            │             │
│  │  ▪ PDFLoader         │  │  ▪ QdrantStore            │             │
│  │  ▪ HTMLLoader        │  │  ▪ ChromaStore (alt)      │             │
│  │  ▪ MarkdownLoader    │  │  ▪ DenseSearch            │             │
│  │  ▪ DOCXLoader        │  │  ▪ SparseSearch (BM25)    │             │
│  │  ▪ RecursiveChunker  │  │  ▪ HybridSearch           │             │
│  │  ▪ SemanticChunker   │  │  ▪ CrossEncoderReranker   │             │
│  │  ▪ BGEM3Embedder     │  └───────────────────────────┘             │
│  │  ▪ SentenceTransf.   │                                           │
│  └─────────────────────┘  ┌───────────────────────────┐             │
│                           │     GENERATION             │             │
│  ┌─────────────────────┐  │  ▪ OllamaLLM               │             │
│  │    PERSISTENCE       │  │  ▪ OpenAICompatibleLLM     │             │
│  │  ▪ SessionRepo       │  │  ▪ PromptTemplates        │             │
│  │  ▪ DocumentRepo      │  └───────────────────────────┘             │
│  └─────────────────────┘                                           │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        INFRAESTRUTURA (Docker)                       │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │  Qdrant   │  │  Ollama  │  │  Redis   │  │  HuggingFace     │    │
│  │  :6333    │  │ :11434   │  │(futuro)  │  │  cache (volume)  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

## Fluxo de Request — /api/ask

```
Usuário → UI → POST /api/ask
  │
  ├─ 1. SessionManager.get_history(session_id)
  │     └─ SQLite ← últimas N mensagens
  │
  ├─ 2. Embedder.embed_query(query)
  │     └─ bge-m3 → vetor 1024 dims
  │
  ├─ 3. RetrievalStrategy.search(query_vector, k=20, filter={user_id, is_public})
  │     ├─ DenseSearch: cosine similarity no Qdrant
  │     ├─ SparseSearch: BM25 no índice de termos
  │     └─ HybridSearch: merge ponderado (alpha=0.5)
  │
  ├─ 4. Reranker.rerank(query, candidates, top_k=5)
  │     └─ Cross-Encoder → scores refinados
  │
  ├─ 5. ConfidenceFilter(top_k, min_score=0.6)
  │     └─ Se score < 0.6: fallback "sem informação"
  │
  ├─ 6. PromptTemplate.render(context=chunks, query=query, history=msgs)
  │     └─ Template: [SISTEMA] + [HISTÓRICO] + [CONTEXTO] + [PERGUNTA]
  │
  ├─ 7. LLM.generate(prompt)
  │     └─ Ollama → llama3.2:8b → resposta + citações
  │
  ├─ 8. SessionManager.add_message(session_id, query, resposta, fontes)
  │     └─ SQLite INSERT
  │
  └─ Response → UI
       ├─ answer: str
       ├─ sources: [{chunk, score, source, page}]
       └─ session_id: str
```

## Fluxo de Ingestion — Upload de Documento

```
Usuário → UI → POST /api/documents (multipart file)
  │
  ├─ 1. Salvar arquivo em data/uploads/{doc_id}/{filename}
  │
  ├─ 2. DocumentLoaderFactory.create(file_extension)
  │     ├─ .pdf   → PDFLoader(pypdf)
  │     ├─ .html  → HTMLLoader(BeautifulSoup)
  │     ├─ .md    → MarkdownLoader
  │     ├─ .docx  → DOCXLoader(python-docx)
  │     └─ .txt   → TextLoader
  │
  ├─ 3. DocumentLoader.load(filepath)
  │     └─ List[Document]  (cada página/seção)
  │
  ├─ 4. Chunker.chunk(documents, chunk_size=512, overlap=0.15)
  │     └─ RecursiveCharacterTextSplitter → List[Chunk]
  │
  ├─ 5. Embedder.embed(chunks.texts)
  │     └─ bge-m3 → List[vector(1024)]
  │
  ├─ 6. VectorStore.add(collection, points=[
  │       { id, vector, payload: {text, source, page, user_id, is_public, timestamp} }
  │     ])
  │     └─ Qdrant → índice atualizado
  │
  └─ Response → UI
       ├─ doc_id: str
       ├─ chunks_count: int
       └─ status: "indexed"
```

## Estrutura de Coleção no Qdrant

```json
{
  "collection_name": "documents",
  "vectors": {
    "size": 1024,
    "distance": "Cosine"
  },
  "payload_schema": {
    "chunk_id": "keyword",
    "doc_id": "keyword",
    "source": "keyword",
    "page": "integer",
    "doc_type": "keyword",
    "user_id": "keyword",
    "is_public": "bool",
    "timestamp": "datetime",
    "text": "text",
    "tags": "list<keyword>"
  }
}
```

## Modelo de Dados — Sessão (SQLite)

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT  -- JSON: { "name": "...", "tags": [...] }
);

CREATE TABLE messages (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    role TEXT CHECK(role IN ('user', 'assistant')),
    content TEXT,
    sources TEXT,  -- JSON: [{ chunk_id, score, source, page }]
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Camadas e Responsabilidades

| Camada | Responsabilidade | Depende de |
|---|---|---|
| **domain** | Entidades puras + contratos (Protocols) | Nada |
| **application** | Orquestração de use cases | domain |
| **infrastructure** | Implementações reais (loaders, Qdrant, Ollama) | domain |
| **presentation** | API HTTP + UI | application |
| **config** | Configuração centralizada (Pydantic) | Nada |

## Regra de Dependência

> Código da camada interna (domain) não importa nada da camada externa.
> Application importa apenas domain.
> Infrastructure implementa ports do domain.
> Presentation depende de application + infrastructure (via Container).

## Estratégia de Substituição de Componentes

Para trocar qualquer componente do sistema:

1. Criar nova classe que implementa o Protocol da domain
2. Registrar na Factory correspondente (ou configurar via .env)
3. Reiniciar o container — sem alterar código de application/presentation

**Exemplo — trocar Qdrant por ChromaDB:**

```python
# infrastructure/retrieval/vector_stores/chroma.py
@VectorStoreFactory.register("chroma")
class ChromaVectorStore:
    """Implementa o protocol VectorStore usando ChromaDB."""
    ...
```

```env
# .env (muda apenas isso)
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

Nenhuma linha de código em `application/` ou `presentation/` precisa mudar.
