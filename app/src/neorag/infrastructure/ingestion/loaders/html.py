from bs4 import BeautifulSoup

from neorag.domain.entities import Document
from neorag.infrastructure.ingestion.loaders.base import DocumentLoader
from neorag.infrastructure.ingestion.loaders.factory import LoaderFactory


@LoaderFactory.register("html")
class HTMLLoader(DocumentLoader):
    async def load(self, source: str) -> list[Document]:
        with open(source, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        title = soup.title.string if soup.title else ""
        body = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
        return [
            Document(
                content=body,
                source=source,
                doc_type="html",
                metadata={"title": title},
            )
        ]
