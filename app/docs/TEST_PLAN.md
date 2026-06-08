# Plano de Testes — 3 Níveis por Feature

## Convenções

- **Unit**: Isolado com mocks. Roda em milissegundos. Sem dependências externas.
- **Integration**: Componentes reais (Qdrant, modelos, banco). Roda em segundos.
- **E2E**: Pipeline completo. Roda em minutos. Requer todos os containers.

```bash
pytest tests/unit/        --cov=src/neorag/domain
pytest tests/integration/ --cov=src/neorag/infrastructure
pytest tests/e2e/         --cov=src/neorag
```

---

## Fase 1 — Infraestrutura Base

### 1.1 Configuração (config.py + Settings)

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_config_defaults` | Valores padrão corretos para todos os submodelos |
| **Unit** | `test_config_from_env` | `os.environ` override funciona via `env_nested_delimiter` |
| **Unit** | `test_config_invalid_type` | `ValidationError` para `embedder.type="inexistente"` |
| **Integration** | `test_config_file_loading` | Carregamento de `.env` arquivo real com valores mock |
| **Integration** | `test_config_export_json` | `model_dump_json()` produz JSON válido com todos os campos |
| **E2E** | `test_health_endpoint` | `GET /health` retorna 200 com `{"status": "ok"}` e versão |

### 1.2 Domain Entities

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_document_creation` | `Document(id, content, metadata)` campos obrigatórios e tipos |
| **Unit** | `test_chunk_creation` | `Chunk` herda `doc_id`, `page` do metadata |
| **Unit** | `test_query_creation` | `Query(text, top_k, filter)` default `top_k=5` |
| **Unit** | `test_session_entity` | `Session(id, messages)` — `add_message()` incrementa lista |
| **Unit** | `test_value_objects` | `ScoredPoint`, `ChunkWithScore` — ordenação por score descendente |
| **Integration** | `test_entity_serialization` | `Document` → JSON → `Document` (roundtrip com Pydantic) |
| **Integration** | `test_session_persistence_roundtrip` | Salvar e carregar Session de SQLite real |
| **E2E** | — | Coberto pelos testes de API |

### 1.3 Domain Ports (Protocols)

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_embedder_protocol` | Classe mock implementa `Embedder` — `isinstance(mock, Embedder)` |
| **Unit** | `test_vector_store_protocol` | Classe mock implementa `VectorStore` — todos os métodos existem |
| **Unit** | `test_llm_protocol` | Classe mock implementa `LLM` — assinaturas corretas |
| **Unit** | `test_chunker_protocol` | Classe mock implementa `Chunker` — retorna `List[Chunk]` |
| **Unit** | `test_loader_protocol` | Classe mock implementa `DocumentLoader` — retorna `List[Document]` |
| **Unit** | `test_reranker_protocol` | Classe mock implementa `Reranker` — assinaturas corretas |
| **Integration** | `test_protocol_runtime_check` | Implementações reais passam `isinstance(real, Protocol)` |
| **E2E** | — | Coberto pelos testes de integração das implementações |

---

## Fase 2 — Pipeline de Ingestion

### 2.1 Document Loaders

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_pdf_loader_initialization` | `PDFLoader()` com path válido, erro com path inválido |
| **Unit** | `test_loader_factory_registry` | `register("pdf")` + `create("pdf")` retorna classe correta |
| **Unit** | `test_loader_factory_unknown` | `create("exe")` levanta `ValueError` |
| **Integration** | `test_pdf_loader_real_file` | Carrega PDF real de teste, retorna `List[Document]` com conteúdo + páginas |
| **Integration** | `test_html_loader_real_file` | Carrega HTML real, extrai texto limpo (sem tags) |
| **Integration** | `test_docx_loader_real_file` | Carrega DOCX real, extrai parágrafos |
| **Integration** | `test_markdown_loader_real_file` | Carrega MD real, preserva headers como metadata |
| **Integration** | `test_loader_unicode_support` | Arquivo com acentos/emoji/UTF-8 é lido corretamente |
| **E2E** | `test_upload_and_load_flow` | Upload via API → arquivo salvo → loader recupera conteúdo |

### 2.2 Chunkers

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_recursive_chunker_empty` | Lista vazia → lista vazia |
| **Unit** | `test_recursive_chunker_small_doc` | Documento menor que chunk_size → 1 chunk |
| **Unit** | `test_recursive_chunker_large_doc` | Documento grande → múltiplos chunks, respeita chunk_size |
| **Unit** | `test_recursive_chunker_overlap` | Overlap de 15% → chunks adjacentes compartilham ~15% tokens |
| **Unit** | `test_recursive_chunker_boundaries` | Separa em `\n\n` antes de `\n` antes de `.` |
| **Unit** | `test_chunker_metadata_propagation` | Metadata do `Document` (source, page) é copiado para cada `Chunk` |
| **Integration** | `test_recursive_vs_semantic_consistency` | Mesmo texto chunked por ambas estratégias — total de chunks diferentes |
| **Integration** | `test_chunker_tiktoken_count` | Contagem de tokens via `tiktoken` coincide com especificado |
| **E2E** | `test_upload_chunk_flow` | Upload PDF → API → chunks corretos no Qdrant |

### 2.3 Embedders

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_embedder_factory_registry` | Cria BGEM3Embedder via Factory com type="bge-m3" |
| **Unit** | `test_embedder_normalization` | Vetor retornado tem norma L2 = 1.0 (após normalize) |
| **Integration** | `test_bge_m3_embed_dimensions` | `embed(["texto"])` retorna vetor com 1024 dimensões |
| **Integration** | `test_bge_m3_embed_multilingual` | Texto em português, inglês e espanhol → vetores válidos |
| **Integration** | `test_bge_m3_embed_batch` | `embed(["a", "b", "c"])` retorna 3 vetores |
| **Integration** | `test_bge_m3_embed_query` | `embed_query("pergunta")` retorna vetor diferente de `embed` |
| **Integration** | `test_embedder_semantic_similarity` | "cachorro" e "cão" têm cosine > 0.7; "cachorro" e "carro" têm < 0.5 |
| **E2E** | `test_chunk_embed_index_flow` | Documento → chunks → embed → Qdrant indexado → search retorna |

### 2.4 Vector Store — Qdrant

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_qdrant_store_initialization` | `QdrantStore(host, port)` com URL inválida levanta ConnectionError |
| **Unit** | `test_qdrant_payload_builder` | Payload com {user_id, source, page, text} formado corretamente |
| **Integration** | `test_qdrant_create_collection` | `create_collection()` com tamanho de vetor correto |
| **Integration** | `test_qdrant_add_and_search` | Adiciona 5 pontos → search → retorna top-k corretos |
| **Integration** | `test_qdrant_filter_by_user` | Adiciona docs user_a e user_b → filtro por user_id retorna só os corretos |
| **Integration** | `test_qdrant_filter_by_public` | Adiciona públicos e privados → filtro is_public=true retorna só públicos |
| **Integration** | `test_qdrant_delete_points` | Adiciona → delete → search não retorna |
| **Integration** | `test_qdrant_search_empty` | Coleção vazia → search retorna lista vazia |
| **E2E** | `test_full_ingestion_to_qdrant` | Upload → load → chunk → embed → Qdrant → search returns |

---

## Fase 3 — Retrieval Pipeline

### 3.1 Search Strategies

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_dense_search_strategy` | Mock Embedder + Mock VectorStore → `search()` chama os métodos na ordem correta |
| **Unit** | `test_sparse_search_strategy` | BM25 com corpus conhecido → scores exatos para query específica |
| **Unit** | `test_hybrid_search_alpha_1` | `alpha=1.0` = apenas dense |
| **Unit** | `test_hybrid_search_alpha_0` | `alpha=0.0` = apenas sparse |
| **Unit** | `test_hybrid_search_alpha_05` | `alpha=0.5` → merge 50/50 com scores normalizados |
| **Integration** | `test_dense_search_real_qdrant` | Documentos reais no Qdrant → busca semântica retorna relevantes |
| **Integration** | `test_sparse_search_bm25_accuracy` | BM25 com termos exatos → precision@5 > 0.8 |
| **Integration** | `test_hybrid_vs_dense_only` | Hybrid recupera docs que dense-only perde (códigos, IDs) |
| **Integration** | `test_search_with_tenant_filter` | Documentos de tenants diferentes → filtro correto |
| **E2E** | `test_search_endpoint` | `POST /api/search` com query → 200 + chunks ranqueados |

### 3.2 Reranker

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_reranker_initialization` | `CrossEncoderReranker(model_name)` — modelo válido |
| **Unit** | `test_reranker_empty_input` | Lista vazia → lista vazia |
| **Unit** | `test_reranker_top_k_limit` | 10 candidatos, `top_k=3` → retorna 3 |
| **Integration** | `test_reranker_scores_ordering` | Query + documentos → scores decrescentes, relevantes no topo |
| **Integration** | `test_reranker_improves_precision` | Precision@5 antes vs depois do reranker (deve melhorar) |
| **Integration** | `test_reranker_consistency` | Mesma entrada → mesma saída (determinístico) |
| **E2E** | `test_retrieve_with_rerank_flow` | Query → retrieve 20 → rerank 5 → resultados corretos |

### 3.3 Confidence Threshold

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_confidence_filter_above` | Todos scores >= 0.6 → mantém todos |
| **Unit** | `test_confidence_filter_below` | Todos scores < 0.6 → lista vazia |
| **Unit** | `test_confidence_filter_mixed` | Mix acima/abaixo → só mantém acima |
| **Integration** | `test_confidence_with_reranker` | Reranker + filter → só resultados com confiança alta |
| **E2E** | `test_low_confidence_fallback` | Query sem docs relevantes → resposta "sem informação" |

---

## Fase 4 — Geração + API

### 4.1 LLM Integration

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_ollama_llm_initialization` | `OllamaLLM(base_url, model)` — URL inválida → ConnectionError |
| **Unit** | `test_llm_factory` | Cria `OllamaLLM` via Factory com type="ollama" |
| **Unit** | `test_llm_factory_unknown` | `type="inexistente"` → ValueError |
| **Integration** | `test_ollama_generate` | `generate("Hello")` → resposta não vazia, sem erros |
| **Integration** | `test_ollama_generate_stream` | `generate_stream("Hello")` → `AsyncIterator` com chunks |
| **Integration** | `test_ollama_temperature_effect` | `temperature=0.0` é determinístico; `temperature=0.7` varia |
| **E2E** | `test_ask_endpoint_llm_response` | `POST /ask` → resposta gerada por LLM, não mock |

### 4.2 Prompt Templates

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_prompt_render_basic` | Template com contexto + pergunta → string renderizada |
| **Unit** | `test_prompt_render_with_history` | Template com histórico de 3 mensagens → inclui todas |
| **Unit** | `test_prompt_render_empty_context` | Contexto vazio → instrução "sem informação" |
| **Unit** | `test_prompt_token_count` | Prompt renderizado não excede `max_tokens` configurado |
| **Unit** | `test_prompt_citation_format` | Citações formatadas corretamente: `[1]`, `[2]` |
| **Integration** | `test_prompt_with_real_context` | Chunks reais do Qdrant → prompt bem formado |
| **E2E** | `test_ask_citations_in_response` | Resposta da API contém referências `[fonte]` |

### 4.3 Session Manager

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_session_create` | `create_session(user_id)` → session com id, created_at |
| **Unit** | `test_session_add_message` | `add_message(session, "user", "text")` → 1 mensagem na lista |
| **Unit** | `test_session_get_history` | `get_history(session, limit=5)` → últimas 5 mensagens |
| **Unit** | `test_session_history_limit` | 20 mensagens, `limit=5` → retorna 5 (não 20) |
| **Unit** | `test_session_not_found` | `get("inexistente")` → None |
| **Integration** | `test_session_sqlite_persistence` | Criar → salvar → carregar → mensagens preservadas |
| **Integration** | `test_session_sqlite_concurrent` | Duas sessões simultâneas → sem dados cruzados |
| **Integration** | `test_session_message_order` | Mensagens em ordem cronológica |
| **E2E** | `test_conversation_flow` | POST /ask (3x) → histórico mantém contexto entre turnos |

### 4.4 API Endpoints

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_ask_request_schema` | Payload inválido → 422 Validation Error |
| **Unit** | `test_document_upload_schema` | Upload sem file → 422 |
| **Integration** | `test_ask_endpoint_success` | POST /ask com query → 200 + answer + sources |
| **Integration** | `test_ask_endpoint_empty_query` | Query vazia → 422 |
| **Integration** | `test_ask_endpoint_with_session` | `session_id` existente → 200 + histórico considerado |
| **Integration** | `test_documents_list_endpoint` | GET /documents → lista de documentos indexados |
| **Integration** | `test_document_delete_endpoint` | DELETE /documents/{id} → 204 + Qdrant sem chunks |
| **Integration** | `test_sessions_list_endpoint` | GET /sessions → lista |
| **E2E** | `test_upload_then_ask_flow` | Upload doc → esperar indexação → perguntar → resposta correta |
| **E2E** | `test_conversation_context_flow` | Pergunta 1 → resposta → Pergunta 2 (referencia P1) → contexto mantido |

---

## Fase 5 — UI Streamlit

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_ui_api_client_format` | `api_client.py` — formata resposta da API corretamente |
| **Unit** | `test_ui_session_state` | `st.session_state` — mensagens são acumuladas |
| **Integration** | `test_ui_upload_component` | Upload de arquivo via `st.file_uploader` → chamada API correta |
| **Integration** | `test_ui_chat_display` | Mensagens de usuário e assistente exibidas estilizadas |
| **Integration** | `test_ui_sources_expander` | Fontes são exibidas em expander |
| **E2E** | `test_user_journey_upload_chat` | Upload → ver mensagem de sucesso → perguntar → ver resposta + fontes |
| **E2E** | `test_user_journey_new_session` | Chat → nova sessão → chat anterior preservado |
| **E2E** | `test_user_journey_public_private` | Modo público vs privado → resultados filtrados corretamente |

---

## Fase 6 — Avaliação e Monitoramento

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_ragas_faithfulness_score` | Resposta fiel ao contexto → score alto |
| **Unit** | `test_ragas_context_precision` | Contexto relevante → score alto |
| **Unit** | `test_ragas_answer_relevancy` | Resposta relevante à pergunta → score alto |
| **Unit** | `test_latency_tracking` | `track_latency("embedding", 0.15)` → log estruturado |
| **Integration** | `test_evaluation_with_sample_queries` | 10 queries com ground truth → relatório de métricas |
| **Integration** | `test_regression_after_index_update` | Adicionar documento → reavaliar → métricas não pioram |
| **E2E** | `test_full_eval_pipeline` | Ingestion → retrieval → geração → RAGAS → relatório JSON |

---

## Fase 7 — Multi-Tenancy + Agentic

| Nível | Teste | O que verifica |
|---|---|---|
| **Unit** | `test_multi_tenant_isolation` | Usuário A não vê docs do Usuário B |
| **Unit** | `test_agentic_query_decomposition` | Query complexa → sub-queries geradas corretamente |
| **Integration** | `test_multi_tenant_overlap` | Usuários diferentes, mesmo documento público → ambos acessam |
| **Integration** | `test_agentic_multi_hop` | Query multi-hop → múltiplos retrievals → resposta sintetizada |
| **Integration** | `test_agentic_confidence_retry` | Confiança baixa → retrieve adicional → resposta melhora |
| **E2E** | `test_multi_tenant_upload_ask` | User A upload privado → User B não vê → User A vê |
| **E2E** | `test_agentic_complex_query` | "Qual produto X e quem era o gerente em 2024" → resposta composta |

---

## Resumo de Quantidade de Testes

| Fase | Unit | Integration | E2E | Total |
|---|---|---|---|---|
| 1 — Infra | 12 | 3 | 1 | 16 |
| 2 — Ingestion | 14 | 14 | 3 | 31 |
| 3 — Retrieval | 9 | 9 | 2 | 20 |
| 4 — Geração + API | 17 | 17 | 4 | 38 |
| 5 — UI | 2 | 3 | 3 | 8 |
| 6 — Avaliação | 4 | 2 | 1 | 7 |
| 7 — Multi/Agentic | 2 | 4 | 2 | 8 |
| **Total** | **60** | **52** | **16** | **128** |
