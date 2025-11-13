"""SQLite-backed store for custom agent tools and research notes."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_toolkit import run_python as core_run_python
from agent_toolkit import run_shell as core_run_shell


def _state_dir() -> Path:
    base = os.environ.get("AGENT_STATE_DIR") or os.path.join("~", ".universal_agent")
    path = Path(base).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError:
        fallback = Path.cwd() / ".agent_state"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


DEFAULT_DB_PATH = Path(
    os.environ.get("AGENT_STATE_DB") or (_state_dir() / "agent_state.sqlite3")
).expanduser()
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _loads_or_empty(blob: Optional[str]) -> Dict[str, Any]:
    if not blob:
        return {}
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


@dataclass(slots=True)
class CustomToolRecord:
    name: str
    description: str
    kind: str
    body: str
    args_schema: Dict[str, Any]
    metadata: Dict[str, Any]
    timeout: int
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResearchNote:
    namespace: str
    key: str
    content: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CustomToolStore:
    """Thread-safe helper around the persistent SQLite database."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path or DEFAULT_DB_PATH).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS custom_tools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    body TEXT NOT NULL,
                    args_schema TEXT,
                    metadata TEXT,
                    timeout INTEGER NOT NULL DEFAULT 120,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS research_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    note_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(namespace, note_key)
                );
                """
            )

    # ---- Custom tool helpers -------------------------------------------------

    def list_tools(self) -> List[CustomToolRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, description, kind, body, args_schema, metadata, timeout, "
                "created_at, updated_at FROM custom_tools ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [self._row_to_tool(row) for row in rows]

    def get_tool(self, name: str) -> Optional[CustomToolRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name, description, kind, body, args_schema, metadata, timeout, "
                "created_at, updated_at FROM custom_tools WHERE name = ?",
                (name,),
            ).fetchone()
        return self._row_to_tool(row) if row else None

    def create_tool(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        body: str,
        args_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> CustomToolRecord:
        if self.get_tool(name):
            raise ValueError(f"Tool '{name}' already exists.")
        return self._write_tool(
            name=name,
            description=description,
            kind=kind,
            body=body,
            args_schema=args_schema or {},
            metadata=metadata or {},
            timeout=timeout or 120,
        )

    def update_tool(
        self,
        name: str,
        *,
        description: Optional[str] = None,
        kind: Optional[str] = None,
        body: Optional[str] = None,
        args_schema: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> CustomToolRecord:
        current = self.get_tool(name)
        if current is None:
            raise ValueError(f"Tool '{name}' was not found.")
        return self._write_tool(
            name=name,
            description=description or current.description,
            kind=kind or current.kind,
            body=body or current.body,
            args_schema=args_schema if args_schema is not None else current.args_schema,
            metadata=metadata if metadata is not None else current.metadata,
            timeout=timeout or current.timeout,
            created_at=current.created_at,
        )

    def delete_tool(self, name: str) -> bool:
        with self._lock, self._connect() as conn:
            result = conn.execute("DELETE FROM custom_tools WHERE name = ?", (name,))
            return result.rowcount > 0

    def run_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        record = self.get_tool(name)
        if record is None:
            raise ValueError(f"Tool '{name}' does not exist.")
        args = arguments or {}
        if record.kind == "shell":
            try:
                command = record.body.format(**args)
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Missing argument for placeholder: {exc}") from exc
            return core_run_shell(command, timeout=record.timeout)
        if record.kind == "python":
            payload = json.dumps(args, ensure_ascii=False)
            code = (
                "import json\n"
                f"params = json.loads({json.dumps(payload)})\n"
                + record.body
            )
            return core_run_python(code, timeout=record.timeout)
        raise ValueError(
            f"Unsupported tool kind '{record.kind}'. Allowed kinds are 'shell' and 'python'."
        )

    def _write_tool(
        self,
        *,
        name: str,
        description: str,
        kind: str,
        body: str,
        args_schema: Dict[str, Any],
        metadata: Dict[str, Any],
        timeout: int,
        created_at: Optional[str] = None,
    ) -> CustomToolRecord:
        now = _utc_timestamp()
        created = created_at or now
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO custom_tools
                    (name, description, kind, body, args_schema, metadata, timeout, created_at, updated_at)
                VALUES
                    (:name, :description, :kind, :body, :args_schema, :metadata, :timeout, :created_at, :updated_at)
                ON CONFLICT(name) DO UPDATE SET
                    description=excluded.description,
                    kind=excluded.kind,
                    body=excluded.body,
                    args_schema=excluded.args_schema,
                    metadata=excluded.metadata,
                    timeout=excluded.timeout,
                    updated_at=excluded.updated_at
                ;
                """,
                {
                    "name": name,
                    "description": description,
                    "kind": kind,
                    "body": body,
                    "args_schema": json.dumps(args_schema or {}),
                    "metadata": json.dumps(metadata or {}),
                    "timeout": int(timeout),
                    "created_at": created,
                    "updated_at": now,
                },
            )
        return self.get_tool(name)  # type: ignore[return-value]

    def _row_to_tool(self, row: sqlite3.Row) -> CustomToolRecord:
        return CustomToolRecord(
            name=row["name"],
            description=row["description"],
            kind=row["kind"],
            body=row["body"],
            args_schema=_loads_or_empty(row["args_schema"]),
            metadata=_loads_or_empty(row["metadata"]),
            timeout=int(row["timeout"] or 120),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ---- Research note helpers ----------------------------------------------

    def list_notes(self, namespace: Optional[str] = None) -> List[ResearchNote]:
        query = (
            "SELECT namespace, note_key, content, metadata, created_at, updated_at "
            "FROM research_notes "
        )
        params: tuple[Any, ...] = ()
        if namespace:
            query += "WHERE namespace = ? "
            params = (namespace,)
        query += "ORDER BY namespace, note_key"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_note(row) for row in rows]

    def get_note(self, namespace: str, key: str) -> Optional[ResearchNote]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT namespace, note_key, content, metadata, created_at, updated_at "
                "FROM research_notes WHERE namespace = ? AND note_key = ?",
                (namespace, key),
            ).fetchone()
        return self._row_to_note(row) if row else None

    def write_note(
        self,
        namespace: str,
        key: str,
        content: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        mode: str = "overwrite",
    ) -> ResearchNote:
        current = self.get_note(namespace, key)
        if current and mode == "append":
            content = current.content + content
            if metadata is None:
                metadata = current.metadata
        now = _utc_timestamp()
        created_at = current.created_at if current else now
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO research_notes
                    (namespace, note_key, content, metadata, created_at, updated_at)
                VALUES
                    (:namespace, :note_key, :content, :metadata, :created_at, :updated_at)
                ON CONFLICT(namespace, note_key) DO UPDATE SET
                    content=excluded.content,
                    metadata=CASE
                        WHEN :metadata IS NOT NULL THEN :metadata
                        ELSE research_notes.metadata
                    END,
                    updated_at=excluded.updated_at
                ;
                """,
                {
                    "namespace": namespace,
                    "note_key": key,
                    "content": content,
                    "metadata": json.dumps(metadata or {}),
                    "created_at": created_at,
                    "updated_at": now,
                },
            )
        return self.get_note(namespace, key)  # type: ignore[return-value]

    def delete_note(self, namespace: str, key: str) -> bool:
        with self._lock, self._connect() as conn:
            result = conn.execute(
                "DELETE FROM research_notes WHERE namespace = ? AND note_key = ?",
                (namespace, key),
            )
            return result.rowcount > 0

    def _row_to_note(self, row: sqlite3.Row) -> ResearchNote:
        return ResearchNote(
            namespace=row["namespace"],
            key=row["note_key"],
            content=row["content"],
            metadata=_loads_or_empty(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


STORE = CustomToolStore()

__all__ = [
    "CustomToolRecord",
    "ResearchNote",
    "CustomToolStore",
    "STORE",
]
