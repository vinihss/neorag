from neorag.application.retrieval import RetrievalService
from neorag.config import Settings
from neorag.domain.ports import Embedder, VectorStore
from neorag.infrastructure.ingestion.embedders.factory import EmbedderFactory
from neorag.infrastructure.retrieval.rerankers.base import Reranker
from neorag.infrastructure.retrieval.vector_stores.factory import VectorStoreFactory


class Container:
    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or Settings()

    @property
    def settings(self) -> Settings:
        return self.config

    def embedder(self) -> Embedder:
        cfg = self.config.embedder
        return EmbedderFactory.create(
            cfg.type,
            model_name=cfg.model_name,
            device=cfg.device,
            normalize=cfg.normalize,
        )

    def vector_store(self) -> VectorStore:
        cfg = self.config.qdrant
        return VectorStoreFactory.create(
            "qdrant",
            host=cfg.host,
            port=cfg.port,
            prefer_grpc=cfg.prefer_grpc,
        )

    def reranker(self) -> Reranker | None:
        return None

    def retrieval_service(self) -> RetrievalService:
        return RetrievalService(
            embedder=self.embedder(),
            vector_store=self.vector_store(),
            config=self.config.retrieval,
            reranker=self.reranker(),
            collection_name=self.config.qdrant.collection_name,
        )
