from neorag.domain.ports import VectorStore


class VectorStoreFactory:
    _registry: dict[str, type[VectorStore]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[VectorStore]) -> type[VectorStore]:
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: object) -> VectorStore:
        if name not in cls._registry:
            msg = f"Unknown vector store: {name}. Available: {list(cls._registry)}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)
