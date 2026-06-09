from neorag.infrastructure.ingestion.chunkers.base import Chunker


class ChunkerFactory:
    _registry: dict[str, type[Chunker]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[Chunker]) -> type[Chunker]:
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: object) -> Chunker:
        if name not in cls._registry:
            msg = f"Unknown chunker: {name}. Available: {list(cls._registry)}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)
