import pypdf

from neorag.domain.entities import Document
from neorag.infrastructure.ingestion.loaders.base import DocumentLoader
from neorag.infrastructure.ingestion.loaders.factory import LoaderFactory


@LoaderFactory.register("pdf")
class PDFLoader(DocumentLoader):
    async def load(self, source: str) -> list[Document]:
        documents: list[Document] = []
        reader = pypdf.PdfReader(source)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append(
                    Document(
                        content=text.strip(),
                        source=source,
                        doc_type="pdf",
                        metadata={"page": i + 1, "total_pages": len(reader.pages)},
                    )
                )
        return documents
