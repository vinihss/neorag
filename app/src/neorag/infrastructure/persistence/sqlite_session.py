import json
from datetime import datetime

import aiosqlite

from neorag.domain.entities import Message, Session
from neorag.domain.ports import SessionRepository
from neorag.domain.value_objects import Source


class SQLiteSessionRepository(SessionRepository):
    def __init__(self, db_path: str = "data/sessions.db") -> None:
        self._db_path = db_path
        self._connection: aiosqlite.Connection | None = None

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._connection is None:
            self._connection = await aiosqlite.connect(self._db_path)
            self._connection.row_factory = aiosqlite.Row
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            await self._connection.commit()
        return self._connection

    async def get(self, session_id: str) -> Session | None:
        conn = await self._get_conn()
        cursor = await conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return None

        messages = await self.get_history(session_id, limit=1000)

        return Session(
            id=row["id"],
            user_id=row["user_id"],
            metadata=json.loads(row["metadata"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            messages=messages,
        )

    async def save(self, session: Session) -> None:
        conn = await self._get_conn()
        await conn.execute(
            """INSERT OR REPLACE INTO sessions (id, user_id, metadata, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                session.id,
                session.user_id,
                json.dumps(session.metadata),
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
            ),
        )
        await conn.commit()

    async def add_message(self, session_id: str, message: Message) -> None:
        conn = await self._get_conn()
        sources_json = json.dumps(
            [
                {
                    "title": s.title,
                    "url": s.url,
                    "page": s.page,
                    "snippet": s.snippet,
                }
                for s in message.sources
            ]
        )
        await conn.execute(
            """INSERT INTO messages (session_id, role, content, sources, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                session_id,
                message.role,
                message.content,
                sources_json,
                message.created_at.isoformat(),
            ),
        )
        await conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), session_id),
        )
        await conn.commit()

    async def get_history(
        self, session_id: str, limit: int = 10
    ) -> list[Message]:
        conn = await self._get_conn()
        cursor = await conn.execute(
            """SELECT * FROM messages WHERE session_id = ?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        messages: list[Message] = []
        for row in reversed(rows):
            sources_data = json.loads(row["sources"])
            sources = [
                Source(
                    title=s.get("title", ""),
                    url=s.get("url", ""),
                    page=s.get("page"),
                    snippet=s.get("snippet", ""),
                )
                for s in sources_data
            ]
            messages.append(
                Message(
                    role=row["role"],
                    content=row["content"],
                    sources=sources,
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
            )
        return messages

    async def list_sessions(
        self, user_id: str | None = None
    ) -> list[Session]:
        conn = await self._get_conn()
        if user_id:
            cursor = await conn.execute(
                "SELECT id FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            )
        else:
            cursor = await conn.execute(
                "SELECT id FROM sessions ORDER BY updated_at DESC"
            )
        rows = await cursor.fetchall()
        sessions: list[Session] = []
        for row in rows:
            session = await self.get(row["id"])
            if session:
                sessions.append(session)
        return sessions
