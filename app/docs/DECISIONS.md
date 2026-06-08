# Decisões Técnicas — NeoRAG

## Stack Tecnológica

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Linguagem | Python 3.12+ | — | Ecossistema maduro para ML/NLP, LangChain, comunidade |
| Package manager | `uv` | pip/poetry/pdm | 10-100x mais rápido, lockfile nativo, `uv sync` reproduzível |
| Web framework | FastAPI | Flask, Django | Assíncrono nativo, Pydantic integrado, performance, OpenAPI automático |
| UI | Streamlit | Gradio, React | Zero boilerplate para ferramentas de dados, stateful fácil, integração Python |
| Orquestração RAG | LangChain | Haystack, manual | Maior ecossistema de loaders/embeddings, flexibilidade, comunidade ativa |

## Embedding

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Modelo primário | BAAI/bge-m3 | all-MiniLM-L6-v2, jina-embeddings-v3 | **Multilíngue** (100+ idiomas), suporte a dense + sparse + multi-vector, #2 no MTEB |
| Framework | sentence-transformers | OpenAI API, Ollama embed | Gratuito, local, sem dependência externa, privacidade total |
| Dimensão | 1024 | 384 (MiniLM), 1536 (OpenAI) | Melhor relação qualidade-performance para multilíngue |
| Normalização | L2 normalize | — | Obrigatório para cosine similarity funcionar corretamente |

**Por que não OpenAI?** Custo por chamada, latência de rede, dados trafegam externamente, vendor lock-in.

**Por que não all-MiniLM-L6-v2?** Apenas inglês. O projeto requer multilíngue (português + outros).

## Vector Database

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Engine primário | Qdrant | ChromaDB, Milvus, FAISS | Open-source, performático (Rust), filtros nativos avançados, API REST+GRPC, Docker simples |
| Coleções | Única com payload filter | Múltiplas coleções | Uma coleção com filtro por `user_id` + `is_public` é mais flexível e econômica |
| Métrica de distância | Cosine | Dot product, L2 | Compatível com embeddings normalizados, range [0, 1] interpretável |

**Por que não FAISS?** Sem persistência nativa, sem filtros por metadados, sem multi-tenancy.

**Por que não Milvus?** Mais complexo de operar (depende de etcd + minIO), overhead para o porte atual.

## Chunking

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Estratégia padrão | RecursiveCharacterTextSplitter | Fixed-size, Semantic | Respeita boundaries naturais (parágrafos, sentenças) |
| Tamanho | 512 tokens | 256, 768, 1024 | Equilíbrio entre precisão e contexto suficiente para resposta |
| Overlap | 15% (≈77 tokens) | 10%, 20% | Empírico: elimina perda de contexto entre chunks sem duplicação excessiva |
| Separadores | `["\n\n", "\n", ".", " ", ""]` | — | Prioriza quebras naturais antes de fallback para caractere |
| Estratégia avançada | SemanticChunker (planejado) | — | Para documentos estruturados, agrupa por tópico antes de chunk |

## Retrieval

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Estratégia | Hybrid Search (dense + sparse) | Dense-only, Sparse-only | Combina semântica (sinônimos, contexto) + termos exatos (códigos, nomes) |
| Alpha (peso dense) | 0.5 configurável | — | Ajustável por domínio via .env |
| Top-K retrieve | 20 | 10, 50 | Suficiente para reranker ter bons candidatos |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | bge-reranker-v2-m3 | Leve (CPU), ganho comprovado de 73% → 89% acurácia |
| Top-K rerank | 5 | 3, 10 | Balanço entre qualidade do contexto e janela de tokens do LLM |
| Confidence threshold | 0.6 | — | Abaixo disso: "Não encontrei informação suficiente" |

**Por que Hybrid?** Dense search sozinho perde matches exatos (códigos, IDs, nomes próprios). Sparse (BM25) sozinho perde semântica. A combinação cobre ambos.

**Por que Reranker é essencial?** Embedding search é aproximado. Cross-Encoder avalia cada par (query, doc) individualmente com precisão muito maior. Custo: O(n) para n candidatos.

## LLM / Geração

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Engine | Ollama | vLLM, TGI | Mais simples de operar, GPU opcional, REST API nativa, Docker oficial |
| Modelo | llama3.2:8b | mistral, phi-3, qwen2.5 | 8B parâmetros = boa qualidade em CPU, multilíngue, licença permissiva |
| Temperatura | 0.1 | 0.0-0.7 | Baixa temperatura = respostas mais determinísticas e factuais |
| Max tokens | 1024 | — | Suficiente para respostas completas sem estourar contexto |
| Modo | Chat (histórico + contexto) | Completion simples | Necessário para conversas multi-turno |

**Por que modelo local?** Privacidade dos dados, sem custo por token, latência previsível, sem vendor lock-in.

**Por que Ollama e não vLLM?** vLLM é mais performático para GPU, mas requer GPU obrigatoriamente. Ollama roda em CPU e GPU, tem API compatível com OpenAI.

## Multi-Tenancy

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Estratégia | Payload filter no Qdrant | Coleções separadas | Sem overhead administrativo, query única com filtro |
| Metadado de acesso | `user_id: str \| None` + `is_public: bool` | — | Documentos sem user_id são públicos; com user_id são privados |
| Isolamento | Application-level (serviço valida) | DB-level | Controle fino sobre permissões, sem depender de feature do vector DB |

**Por que payload filter?** Uma única coleção é mais simples de gerenciar, fazer backup, e escalar. O filtro de payload do Qdrant é eficiente (índice por campo).

## Sessão e Persistência

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Storage de sessões | SQLite via aiosqlite | Redis, PostgreSQL | Zero dependências extras, suficiente para sessões, transacional |
| Formato | JSON column para mensagens | Tabela separada | Simplicidade de schema, sem joins, flexível |
| Cleanup | TTL de 7 dias (configurável) | — | Evita crescimento infinito do banco |

## Infraestrutura

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Container runtime | Docker Compose | Kubernetes | Simplicidade, ambiente dev/prod homogêneo, orquestração local |
| Volumes | Bind mounts para dados | Named volumes | Transparência, backup fácil, acesso direto aos arquivos |
| Healthcheck | Endpoint /health + depends_on | — | Ordem de inicialização correta (qdrant → ollama → app) |
| Model cache | Volume para ~/.cache/huggingface | — | Evita redownload de modelos a cada restart |

## Logging e Observabilidade

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Logger | structlog | loguru, logging | Structured JSON logging, context binding, integração com OpenTelemetry |
| Métricas | Prometheus client (planejado) | — | Latência por etapa, taxa de erro, custo por query |
| Rastreamento | OpenTelemetry (planejado) | — | Tracing distribuído para debugging de pipelines |

## Testes

| Decisão | Escolha | Alternativas | Motivação |
|---|---|---|---|
| Framework | pytest + pytest-asyncio | unittest | Sintaxe concisa, fixtures poderosas, suporte nativo a async |
| Mock | unittest.mock / pytest-mock | — | Já incluso no stdlib |
| Cobertura | pytest-cov | — | Relatório de cobertura por módulo |
| Qualidade | ruff + mypy | flake8, black, pylint | Linter + formatador (ruff) + type checker (mypy) |
| Pré-commit | pre-commit hooks | — | Automatiza ruff + mypy antes de cada commit |

---

## Arquitetura de Testes por Feature

Cada feature tem **3 níveis de teste** documentados abaixo. A estrutura de diretórios reflete essa organização:

```
tests/
├── unit/                    # Testes isolados (mocks)
│   ├── domain/              # Entidades e value objects
│   ├── application/         # Serviços com dependências mockadas
│   └── infrastructure/      # Implementações mockadas
├── integration/             # Componentes reais (DB, modelos)
│   └── ...                  # Uma bateria por módulo
└── e2e/                     # Pipeline completo
    └── ...                  # Fluxos ponta-a-ponta
```
