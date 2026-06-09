from neorag.infrastructure.ingestion.loaders.base import DocumentLoader


class LoaderFactory:
    _registry: dict[str, type[DocumentLoader]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[DocumentLoader]) -> type[DocumentLoader]:
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: object) -> DocumentLoader:
        if name not in cls._registry:
            msg = f"Unknown loader: {name}. Available: {list(cls._registry)}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)

    @classmethod
    def registered(cls) -> list[str]:
        return list(cls._registry)
