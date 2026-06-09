from neorag.infrastructure.retrieval.strategies.base import RetrievalStrategy


class RetrievalStrategyFactory:
    _registry: dict[str, type[RetrievalStrategy]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[RetrievalStrategy]) -> type[RetrievalStrategy]:
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: object) -> RetrievalStrategy:
        if name not in cls._registry:
            msg = f"Unknown retrieval strategy: {name}. Available: {list(cls._registry)}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)
