from neorag.config import Settings


class Container:
    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or Settings()

    @property
    def settings(self) -> Settings:
        return self.config
