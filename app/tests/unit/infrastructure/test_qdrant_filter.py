
from neorag.infrastructure.retrieval.vector_stores.qdrant import QdrantStore


class TestQdrantFilter:
    def test_build_simple_filter(self):
        store = QdrantStore(host="localhost", port=6333)
        f = store._build_filter({"is_public": True})
        assert f.must is not None
        assert len(f.must) == 1
        assert f.should is None

    def test_build_filter_with_should(self):
        store = QdrantStore(host="localhost", port=6333)
        f = store._build_filter(
            {
                "doc_type": "pdf",
                "_should": [
                    {"user_id": "u1"},
                    {"is_public": True},
                ],
            }
        )
        assert f.must is not None
        assert len(f.must) == 1
        assert f.should is not None
        assert len(f.should) == 2

    def test_build_filter_should_only(self):
        store = QdrantStore(host="localhost", port=6333)
        f = store._build_filter(
            {
                "_should": [
                    {"user_id": "u1"},
                    {"is_public": True},
                ],
            }
        )
        assert f.should is not None
        assert len(f.should) == 2
