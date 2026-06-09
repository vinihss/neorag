import pytest
import tempfile
import os

from neorag.domain.entities import Message, Session
from neorag.domain.value_objects import Source
from neorag.infrastructure.persistence.sqlite_session import SQLiteSessionRepository


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def repo(db_path):
    return SQLiteSessionRepository(db_path=db_path)


@pytest.mark.asyncio
class TestSQLiteSessionRepository:
    async def test_save_and_get_session(self, repo):
        session = Session(id="s1", user_id="u1")
        await repo.save(session)

        retrieved = await repo.get("s1")
        assert retrieved is not None
        assert retrieved.id == "s1"
        assert retrieved.user_id == "u1"

    async def test_get_nonexistent_session(self, repo):
        result = await repo.get("nonexistent")
        assert result is None

    async def test_add_message(self, repo):
        session = Session(id="s2", user_id="u1")
        await repo.save(session)

        msg = Message(role="user", content="hello")
        await repo.add_message("s2", msg)

        history = await repo.get_history("s2")
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "hello"

    async def test_add_message_with_sources(self, repo):
        session = Session(id="s3", user_id="u1")
        await repo.save(session)

        msg = Message(
            role="assistant",
            content="answer",
            sources=[Source(title="doc1", url="/path/to/doc", page=1, snippet="some text")],
        )
        await repo.add_message("s3", msg)

        history = await repo.get_history("s3")
        assert len(history) == 1
        assert len(history[0].sources) == 1
        assert history[0].sources[0].title == "doc1"

    async def test_get_history_limit(self, repo):
        session = Session(id="s4", user_id="u1")
        await repo.save(session)

        for i in range(5):
            await repo.add_message("s4", Message(role="user", content=f"msg{i}"))

        history = await repo.get_history("s4", limit=3)
        assert len(history) == 3

    async def test_list_sessions(self, repo):
        await repo.save(Session(id="a1", user_id="u1"))
        await repo.save(Session(id="a2", user_id="u2"))
        await repo.save(Session(id="a3", user_id="u1"))

        all_sessions = await repo.list_sessions()
        assert len(all_sessions) == 3

        user_sessions = await repo.list_sessions(user_id="u1")
        assert len(user_sessions) == 2

    async def test_update_session_on_add_message(self, repo):
        session = Session(id="s5", user_id="u1")
        await repo.save(session)
        original = await repo.get("s5")

        await repo.add_message("s5", Message(role="user", content="hi"))
        updated = await repo.get("s5")

        assert updated is not None
        assert updated.updated_at >= original.updated_at
