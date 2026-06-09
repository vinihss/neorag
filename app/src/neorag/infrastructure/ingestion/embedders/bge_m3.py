from sentence_transformers import SentenceTransformer

from neorag.domain.ports import Embedder
from neorag.infrastructure.ingestion.embedders.factory import EmbedderFactory


@EmbedderFactory.register("bge-m3")
class BGEM3Embedder(Embedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize: bool = True,
    ) -> None:
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize
        self._dimensions = self._model.get_sentence_embedding_dimension()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=self._normalize)
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(
            text, normalize_embeddings=self._normalize
        )
        return embedding.tolist()
