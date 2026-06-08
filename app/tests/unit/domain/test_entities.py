from uuid import UUID
from datetime import datetime

from neorag.domain.entities import (
    Document,
    Chunk,
    Query,
    Answer,
    Message,
    Session,
)
from neorag.domain.value_objects import ScoredChunk, Source


class TestDocument:
    def test_create_with_defaults(self):
        doc = Document()
        assert doc.id is not None
        assert doc.content == ""
        assert doc.metadata == {}
        assert doc.source == ""
        assert doc.doc_type == ""

    def test_create_with_values(self):
        doc = Document(
            content="test content",
            metadata={"key": "value"},
            source="file.pdf",
            doc_type="pdf",
        )
        assert doc.content == "test content"
        assert doc.metadata == {"key": "value"}
        assert doc.source == "file.pdf"
        assert doc.doc_type == "pdf"

    def test_add_metadata(self):
        doc = Document()
        doc.add_metadata("key1", "value1")
        assert doc.metadata["key1"] == "value1"

    def test_id_is_uuid(self):
        doc = Document()
        UUID(doc.id)  # raises ValueError if invalid


class TestChunk:
    def test_create_with_defaults(self):
        chunk = Chunk()
        assert chunk.id is not None
        assert chunk.document_id == ""
        assert chunk.content == ""

    def test_create_with_values(self):
        chunk = Chunk(
            document_id="doc-1",
            content="chunk content",
            page=3,
            source="file.pdf",
            metadata={"section": "intro"},
        )
        assert chunk.document_id == "doc-1"
        assert chunk.content == "chunk content"
        assert chunk.page == 3
        assert chunk.source == "file.pdf"
        assert chunk.metadata == {"section": "intro"}


class TestQuery:
    def test_create_with_defaults(self):
        q = Query()
        assert q.text == ""
        assert q.top_k == 5
        assert q.filter == {}
        assert q.user_id is None

    def test_create_with_values(self):
        q = Query(text="test", top_k=10, filter={"is_public": True}, user_id="user-1")
        assert q.text == "test"
        assert q.top_k == 10
        assert q.filter == {"is_public": True}
        assert q.user_id == "user-1"


class TestAnswer:
    def test_create_with_defaults(self):
        a = Answer()
        assert a.text == ""
        assert a.sources == []
        assert a.session_id == ""
        assert a.confidence == 0.0

    def test_create_with_sources(self):
        sources = [ScoredChunk(chunk_id="c1", score=0.95)]
        a = Answer(text="resposta", sources=sources, session_id="s1", confidence=0.9)
        assert a.text == "resposta"
        assert len(a.sources) == 1
        assert a.sources[0].score == 0.95


class TestMessage:
    def test_create(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert isinstance(msg.created_at, datetime)

    def test_with_sources(self):
        sources = [Source(title="doc1", url="/path")]
        msg = Message(role="assistant", content="answer", sources=sources)
        assert len(msg.sources) == 1
        assert msg.sources[0].title == "doc1"


class TestSession:
    def test_create_with_defaults(self):
        s = Session()
        assert s.id is not None
        assert s.messages == []
        assert isinstance(s.created_at, datetime)

    def test_add_message(self):
        s = Session(user_id="user-1")
        msg = Message(role="user", content="hello")
        s.add_message(msg)
        assert len(s.messages) == 1
        assert s.messages[0].content == "hello"

    def test_last_messages(self):
        s = Session()
        for i in range(5):
            s.add_message(Message(role="user", content=f"msg-{i}"))
        last = s.last_messages(2)
        assert len(last) == 2
        assert last[-1].content == "msg-4"

    def test_last_messages_limit_exceeds(self):
        s = Session()
        for i in range(3):
            s.add_message(Message(role="user", content=f"msg-{i}"))
        last = s.last_messages(10)
        assert len(last) == 3

    def test_updated_at_on_add(self):
        s = Session()
        original = s.updated_at
        s.add_message(Message(role="user", content="hello"))
        assert s.updated_at >= original
