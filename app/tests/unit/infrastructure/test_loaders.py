import pytest
from pathlib import Path

from neorag.domain.entities import Document
from neorag.infrastructure.ingestion.loaders.factory import LoaderFactory
from neorag.infrastructure.ingestion.loaders import (
    PDFLoader,
    HTMLLoader,
    MarkdownLoader,
    TextLoader,
)

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


class TestLoaderFactory:
    def test_registered_loaders(self):
        assert "pdf" in LoaderFactory.registered()
        assert "html" in LoaderFactory.registered()
        assert "markdown" in LoaderFactory.registered()
        assert "text" in LoaderFactory.registered()

    def test_create_pdf(self):
        loader = LoaderFactory.create("pdf")
        assert isinstance(loader, PDFLoader)

    def test_create_html(self):
        loader = LoaderFactory.create("html")
        assert isinstance(loader, HTMLLoader)

    def test_create_markdown(self):
        loader = LoaderFactory.create("markdown")
        assert isinstance(loader, MarkdownLoader)

    def test_create_text(self):
        loader = LoaderFactory.create("text")
        assert isinstance(loader, TextLoader)

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown loader"):
            LoaderFactory.create("unknown")


class TestPDFLoader:
    @pytest.mark.asyncio
    async def test_load_pdf(self):
        loader = PDFLoader()
        docs = await loader.load(str(FIXTURES / "sample.pdf"))
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert isinstance(docs[0], Document)
        assert docs[0].doc_type == "pdf"
        assert docs[0].source.endswith("sample.pdf")
        assert "page" in docs[0].metadata


class TestHTMLLoader:
    @pytest.mark.asyncio
    async def test_load_html(self):
        loader = HTMLLoader()
        docs = await loader.load(str(FIXTURES / "sample.html"))
        assert len(docs) == 1
        assert docs[0].doc_type == "html"
        assert "Título Principal" in docs[0].content
        assert docs[0].metadata.get("title") == "Página de Teste"

    @pytest.mark.asyncio
    async def test_load_html_removes_tags(self):
        loader = HTMLLoader()
        docs = await loader.load(str(FIXTURES / "sample.html"))
        assert "<h1>" not in docs[0].content
        assert "<p>" not in docs[0].content


class TestMarkdownLoader:
    @pytest.mark.asyncio
    async def test_load_markdown(self):
        loader = MarkdownLoader()
        docs = await loader.load(str(FIXTURES / "sample.md"))
        assert len(docs) == 1
        assert docs[0].doc_type == "markdown"
        assert "RAG" in docs[0].content

    @pytest.mark.asyncio
    async def test_load_empty_markdown(self):
        loader = MarkdownLoader()
        docs = await loader.load(str(FIXTURES / "empty.md"))
        assert len(docs) == 1
        assert docs[0].content == ""


class TestTextLoader:
    @pytest.mark.asyncio
    async def test_load_text(self):
        loader = TextLoader()
        docs = await loader.load(str(FIXTURES / "sample.txt"))
        assert len(docs) == 1
        assert docs[0].doc_type == "text"
        assert "acentuação" in docs[0].content

    @pytest.mark.asyncio
    async def test_load_text_utf8(self):
        loader = TextLoader()
        docs = await loader.load(str(FIXTURES / "sample.txt"))
        assert "ção" in docs[0].content
