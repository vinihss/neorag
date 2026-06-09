from neorag.infrastructure.generation.llms.base import LLM


class LLMFactory:
    _registry: dict[str, type[LLM]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(klass: type[LLM]) -> type[LLM]:
            cls._registry[name] = klass
            return klass
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: object) -> LLM:
        if name not in cls._registry:
            msg = f"Unknown LLM: {name}. Available: {list(cls._registry)}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)
