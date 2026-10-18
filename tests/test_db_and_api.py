"""
Tests for persistence and the HTTP surface.

The API tests use a stub container, so no models load and no network is
touched: what is being verified is the wire contract the frontend depends
on, not the pipeline behind it.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from vectora.config import QueryType, SearchMode
from vectora.db import ChatRepository
from vectora.models import Answer, Citation, Grounding, Usage

# ── repository ────────────────────────────────────────────────────────────


@pytest.fixture
def repo(tmp_path) -> ChatRepository:
    r = ChatRepository(tmp_path / "chats.db")
    yield r
    r.close()


class TestRepository:
    def test_session_lifecycle(self, repo):
        session = repo.create_session()
        assert repo.get_session(session.id) is not None
        assert repo.rename_session(session.id, "Renamed")
        assert repo.get_session(session.id).title == "Renamed"
        assert repo.delete_session(session.id)
        assert repo.get_session(session.id) is None

    def test_exchange_is_atomic_and_titles_the_session(self, repo):
        session = repo.create_session()
        repo.add_exchange(
            session.id,
            "What is the LOD of the assay?",
            "It is 2.5 ng/mL [1].",
            citations=[
                Citation(index=1, chunk_id="c1", paper_id="p1", title="Paper One", page=4)
            ],
        )
        messages = repo.get_messages(session.id)
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].citations[0].title == "Paper One"
        assert repo.get_session(session.id).title.startswith("What is the LOD")

    def test_existing_title_is_not_overwritten(self, repo):
        session = repo.create_session()
        repo.rename_session(session.id, "My title")
        repo.add_exchange(session.id, "a question", "an answer")
        assert repo.get_session(session.id).title == "My title"

    def test_history_is_chronological_and_limited(self, repo):
        session = repo.create_session()
        for i in range(10):
            repo.add_exchange(session.id, f"q{i}", f"a{i}")
        history = repo.get_history(session.id, 4)
        assert len(history) == 4
        assert history[0]["content"] == "q8"
        assert history[-1]["content"] == "a9"

    def test_deleting_a_session_removes_its_messages(self, repo):
        session = repo.create_session()
        repo.add_exchange(session.id, "q", "a")
        repo.delete_session(session.id)
        assert repo.get_messages(session.id) == []

    def test_foreign_keys_are_enforced(self, repo):
        """SQLite has them off by default, so the original schema's
        REFERENCES clause did nothing."""
        with pytest.raises(sqlite3.IntegrityError):
            repo.add_message("no-such-session", "user", "orphan")

    def test_naive_timestamps_from_the_old_schema_are_readable(self, repo, tmp_path):
        session = repo.create_session()
        conn = sqlite3.connect(tmp_path / "chats.db")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
            (session.id, "user", "legacy row", datetime.utcnow().isoformat()),
        )
        conn.commit()
        conn.close()
        messages = repo.get_messages(session.id)
        assert messages[0].created_at.tzinfo is timezone.utc

    def test_legacy_citation_rows_are_tolerated(self, repo, tmp_path):
        """Rows written by the old code lack `index` and `chunk_id`."""
        import json

        session = repo.create_session()
        conn = sqlite3.connect(tmp_path / "chats.db")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, citations, created_at) "
            "VALUES (?,?,?,?,?)",
            (
                session.id,
                "assistant",
                "old answer",
                json.dumps([{"title": "T", "authors": "A", "year": "2020", "page": 3}]),
                "2026-03-18T10:07:17.042589",
            ),
        )
        conn.commit()
        conn.close()
        messages = repo.get_messages(session.id)
        assert messages[0].citations[0].title == "T"

    def test_prune_removes_only_empty_sessions(self, repo):
        used = repo.create_session()
        repo.add_exchange(used.id, "q", "a")
        repo.create_session()
        repo.create_session()
        assert repo.prune_empty_sessions() == 2
        assert len(repo.list_sessions()) == 1

    def test_migration_is_idempotent(self, tmp_path):
        path = tmp_path / "twice.db"
        ChatRepository(path).close()
        ChatRepository(path).close()  # must not raise


# ── API ───────────────────────────────────────────────────────────────────


class _StubEngine:
    def __init__(self) -> None:
        self.retriever = None

    def info(self) -> dict:
        return {"stub": True}

    async def answer(self, query, **kwargs):
        return Answer(
            query=query,
            answer="A grounded answer [1].",
            citations=[
                Citation(index=1, chunk_id="c1", paper_id="p1", title="Paper One", page=4)
            ],
            query_type=QueryType.FACTUAL,
            mode=kwargs.get("mode") or SearchMode.BALANCED,
            grounding=Grounding.from_score(95, summary="All claims supported."),
            usage=Usage(prompt_tokens=100, completion_tokens=20, calls=1, cost_usd=0.0001),
        )

    async def stream(self, query, **kwargs):
        yield {"type": "status", "stage": "retrieving"}
        yield {"type": "sources", "count": 1, "sources": []}
        yield {"type": "token", "text": "A grounded "}
        yield {"type": "token", "text": "answer [1]."}
        answer = await self.answer(query, **kwargs)
        yield {"type": "done", **answer.wire()}


@pytest.fixture
def client(settings, tmp_path):
    from vectora.api.app import create_app
    from vectora.api.deps import Container
    from vectora.cache import Cache
    from vectora.corpus import CorpusContext
    from vectora.store import QdrantStore, SparseIndex

    app = create_app(settings, warm=False)
    repository = ChatRepository(tmp_path / "api.db")

    # Attaching the container before startup means the real one is never
    # built, so no models load and nothing touches the network.
    app.state.container = Container(
        settings=settings,
        corpus=CorpusContext.build(settings),
        cache=Cache(tmp_path / "c.db", 60, enabled=False),
        store=QdrantStore(settings, client=object()),
        sparse=SparseIndex(settings),
        encoder=None,  # type: ignore[arg-type]
        reranker=None,  # type: ignore[arg-type]
        repository=repository,
        engine=_StubEngine(),  # type: ignore[arg-type]
        llm=None,
        startup_warnings=[],
    )
    with TestClient(app) as c:
        yield c
    repository.close()


class TestApi:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_request_id_header_is_always_present(self, client):
        assert client.get("/health").headers["X-Request-ID"]

    def test_session_crud(self, client):
        created = client.post("/sessions")
        assert created.status_code == 201
        sid = created.json()["id"]

        listing = client.get("/sessions")
        assert any(s["id"] == sid for s in listing.json())

        renamed = client.patch(f"/sessions/{sid}", json={"title": "Renamed"})
        assert renamed.status_code == 200

        assert client.delete(f"/sessions/{sid}").status_code == 200
        assert client.get(f"/sessions/{sid}/messages").status_code == 404

    def test_chat_wire_shape_matches_the_frontend(self, client):
        """The shipped index.html reads exactly these keys."""
        sid = client.post("/sessions").json()["id"]
        response = client.post(
            f"/sessions/{sid}/chat", json={"query": "What is the LOD?", "mode": "balanced"}
        )
        assert response.status_code == 200
        body = response.json()
        for key in ("answer", "citations", "faithfulness", "query_type", "fallback"):
            assert key in body
        assert body["faithfulness"]["verdict"] == "FAITHFUL"
        assert body["citations"][0]["title"] == "Paper One"

    def test_chat_persists_the_exchange(self, client):
        sid = client.post("/sessions").json()["id"]
        client.post(f"/sessions/{sid}/chat", json={"query": "A question?"})
        messages = client.get(f"/sessions/{sid}/messages").json()
        assert [m["role"] for m in messages] == ["user", "assistant"]

    def test_chat_on_missing_session_is_404(self, client):
        response = client.post("/sessions/nope/chat", json={"query": "hi"})
        assert response.status_code == 404
        assert "request_id" in response.json()

    def test_blank_query_is_rejected(self, client):
        sid = client.post("/sessions").json()["id"]
        response = client.post(f"/sessions/{sid}/chat", json={"query": "   "})
        assert response.status_code == 422
        assert response.json()["error"] == "Invalid request"

    def test_bad_year_filter_is_rejected(self, client):
        sid = client.post("/sessions").json()["id"]
        response = client.post(f"/sessions/{sid}/chat", json={"query": "q", "year": "20"})
        assert response.status_code == 422

    def test_bad_mode_is_rejected(self, client):
        sid = client.post("/sessions").json()["id"]
        response = client.post(f"/sessions/{sid}/chat", json={"query": "q", "mode": "turbo"})
        assert response.status_code == 422

    def test_stream_emits_tokens_then_done(self, client):
        sid = client.post("/sessions").json()["id"]
        with client.stream(
            "POST", f"/sessions/{sid}/chat/stream", json={"query": "A question?"}
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
        assert '"type": "token"' in body or '"type":"token"' in body
        assert '"type": "done"' in body or '"type":"done"' in body
        assert body.rstrip().endswith("[DONE]")

    def test_export_with_no_messages_is_400(self, client):
        sid = client.post("/sessions").json()["id"]
        assert client.get(f"/sessions/{sid}/export/pdf").status_code == 400
