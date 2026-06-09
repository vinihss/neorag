from neorag.domain.entities import Document
from neorag.infrastructure.ingestion.loaders.base import DocumentLoader
from neorag.infrastructure.ingestion.loaders.factory import LoaderFactory


@LoaderFactory.register("markdown")
class MarkdownLoader(DocumentLoader):
    async def load(self, source: str) -> list[Document]:
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
        return [
            Document(
                content=content,
                source=source,
                doc_type="markdown",
            )
        ]
