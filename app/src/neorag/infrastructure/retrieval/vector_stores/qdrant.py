from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from neorag.domain.ports import VectorStore
from neorag.domain.value_objects import ScoredChunk
from neorag.infrastructure.retrieval.vector_stores.factory import VectorStoreFactory


@VectorStoreFactory.register("qdrant")
class QdrantStore(VectorStore):
    def __init__(
        self,
        host: str = "qdrant",
        port: int = 6333,
        prefer_grpc: bool = True,
    ) -> None:
        self._client = AsyncQdrantClient(
            host=host, port=port, prefer_grpc=prefer_grpc
        )

    async def collection_exists(self, collection: str) -> bool:
        result = await self._client.collection_exists(collection)
        return result

    async def create_collection(
        self, collection: str, vector_size: int
    ) -> None:
        await self._client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size, distance=Distance.COSINE
            ),
        )

    async def add(
        self, collection: str, points: list[dict[str, Any]]
    ) -> None:
        qdrant_points = [
            PointStruct(
                id=p.get("id"),
                vector=p["vector"],
                payload={
                    k: v for k, v in p.items() if k not in ("id", "vector")
                },
            )
            for p in points
        ]
        await self._client.upsert(
            collection_name=collection, points=qdrant_points
        )

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        k: int = 10,
        filter_: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        qdrant_filter = self._build_filter(filter_) if filter_ else None
        results = await self._client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=k,
            query_filter=qdrant_filter,
        )
        return [
            ScoredChunk(
                chunk_id=str(r.id),
                score=r.score,
                text=r.payload.get("text", ""),
                source=r.payload.get("source", ""),
                page=r.payload.get("page"),
                metadata=dict(r.payload),
            )
            for r in results
        ]

    async def delete(
        self, collection: str, point_ids: list[str]
    ) -> None:
        await self._client.delete(
            collection_name=collection,
            points_selector=point_ids,
        )

    def _build_filter(
        self, filter_dict: dict[str, Any]
    ) -> Filter:
        conditions: list[FieldCondition] = []
        for key, value in filter_dict.items():
            if isinstance(value, dict):
                if "gte" in value or "lte" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(
                                gte=value.get("gte"),
                                lte=value.get("lte"),
                            ),
                        )
                    )
            else:
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                )
        return Filter(must=conditions)
