import pytest

from neorag.infrastructure.ingestion.embedders.factory import EmbedderFactory


class TestEmbedderFactory:
    def test_registered_embedders(self):
        assert "bge-m3" in EmbedderFactory._registry
        assert "sentence-transformer" in EmbedderFactory._registry

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown embedder"):
            EmbedderFactory.create("unknown")


@pytest.mark.skip(reason="Requires model download (~2GB)")
class TestBGEM3Embedder:
    @pytest.mark.asyncio
    async def test_embed_dimensions(self):
        embedder = EmbedderFactory.create("bge-m3")
        vectors = await embedder.embed(["hello"])
        assert len(vectors) == 1
        assert embedder.dimensions == 1024

    @pytest.mark.asyncio
    async def test_embed_query(self):
        embedder = EmbedderFactory.create("bge-m3")
        vector = await embedder.embed_query("test query")
        assert len(vector) == 1024

    @pytest.mark.asyncio
    async def test_embed_multiple(self):
        embedder = EmbedderFactory.create("bge-m3")
        vectors = await embedder.embed(["a", "b", "c"])
        assert len(vectors) == 3


@pytest.mark.skip(reason="Requires model download")
class TestSentenceTransformerEmbedder:
    @pytest.mark.asyncio
    async def test_embed(self):
        embedder = EmbedderFactory.create("sentence-transformer")
        vectors = await embedder.embed(["test"])
        assert len(vectors) == 1
