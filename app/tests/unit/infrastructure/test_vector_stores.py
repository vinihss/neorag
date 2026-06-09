import pytest

from neorag.infrastructure.retrieval.vector_stores.factory import VectorStoreFactory


class TestVectorStoreFactory:
    def test_registered_stores(self):
        assert "qdrant" in VectorStoreFactory._registry

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown vector store"):
            VectorStoreFactory.create("unknown")


@pytest.mark.skip(reason="Requires Qdrant running on localhost:6333")
class TestQdrantStore:
    @pytest.mark.asyncio
    async def test_collection_lifecycle(self):
        store = VectorStoreFactory.create("qdrant", host="localhost", port=6333)
        name = "test_collection"

        exists = await store.collection_exists(name)
        if not exists:
            await store.create_collection(name, vector_size=4)

        assert await store.collection_exists(name)

        points = [
            {"id": "1", "vector": [0.1, 0.2, 0.3, 0.4], "text": "doc1"},
            {"id": "2", "vector": [0.5, 0.6, 0.7, 0.8], "text": "doc2"},
        ]
        await store.add(name, points)

        results = await store.search(name, [0.1, 0.2, 0.3, 0.4], k=2)
        assert len(results) == 2

        await store.delete(name, ["1"])
        results = await store.search(name, [0.1, 0.2, 0.3, 0.4], k=2)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_filter(self):
        store = VectorStoreFactory.create("qdrant", host="localhost", port=6333)
        name = "test_filter"

        if not await store.collection_exists(name):
            await store.create_collection(name, vector_size=4)

        points = [
            {"id": "f1", "vector": [0.1, 0.1, 0.1, 0.1], "text": "public", "is_public": True},
            {"id": "f2", "vector": [0.1, 0.1, 0.1, 0.1], "text": "private", "is_public": False},
        ]
        await store.add(name, points)

        results = await store.search(
            name, [0.1, 0.1, 0.1, 0.1], k=5, filter_={"is_public": True}
        )
        assert len(results) == 1
        assert results[0].text == "public"

    @pytest.mark.asyncio
    async def test_search_empty(self):
        store = VectorStoreFactory.create("qdrant", host="localhost", port=6333)
        results = await store.search("nonexistent", [0.1] * 4, k=5)
        assert results == []
