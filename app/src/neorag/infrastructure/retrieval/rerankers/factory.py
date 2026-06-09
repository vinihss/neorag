from neorag.infrastructure.retrieval.rerankers.base import Reranker


class RerankerFactory:
    _registry: dict[str, type[Reranker]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[Reranker]) -> type[Reranker]:
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: object) -> Reranker:
        if name not in cls._registry:
            msg = f"Unknown reranker: {name}. Available: {list(cls._registry)}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)
