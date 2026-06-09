from neorag.domain.ports import Embedder


class EmbedderFactory:
    _registry: dict[str, type[Embedder]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[Embedder]) -> type[Embedder]:
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: object) -> Embedder:
        if name not in cls._registry:
            msg = f"Unknown embedder: {name}. Available: {list(cls._registry)}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)
