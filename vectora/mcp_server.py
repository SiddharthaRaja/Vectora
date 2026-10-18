"""
Vectora as an MCP server.

Exposes the corpus as tools an MCP client can call, so an assistant can
search these documents the way it would any other tool, rather than the
corpus being reachable only through this project's own UI.

Four tools, and the split between them is the design:

* ``search_corpus`` returns passages with their citations and scores. No
  generation, no cost beyond embedding. This is the one a client should
  normally use, because it lets the *client's* model reason over the
  passages with its own context and its own judgement.
* ``answer_question`` runs the full pipeline and returns Vectora's own cited
  answer. Useful when the client wants the grounding check and verified
  quotes rather than raw passages, but it spends an OpenAI call and puts a
  second model between the client and the sources.
* ``list_corpora`` and ``corpus_info`` let a client discover what is
  available and decide whether this corpus can answer at all, instead of
  searching hopefully.

Implemented against the MCP stdio protocol directly over JSON-RPC. The
official SDK would be fewer lines, but it would be another dependency in
the main install path for something that is a thin adapter over an API that
already exists — and writing it out makes the protocol legible to anyone
reading the file, which matters more here.

Run it with::

    vectora mcp

and point a client at that command. It speaks JSON-RPC 2.0 over stdin and
stdout, one message per line, so **nothing may ever be printed to stdout**
except protocol frames: logging goes to stderr, which is why this module
forces that before anything else happens.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from vectora import __version__
from vectora.config import SearchMode, Settings
from vectora.observability import get_logger, request_context

log = get_logger(__name__)

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "vectora"

#: Cap on passages returned in one call. A client's context is not ours to
#: fill, and twenty long chunks is already a lot of tokens.
MAX_RESULTS = 20
MAX_CONTENT_CHARS = 2000


def _tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_corpus",
            "description": (
                "Search the document corpus and return the matching passages "
                "with citations. Does not generate an answer — use this when "
                "you want to read the sources and reason over them yourself. "
                "Hybrid dense + lexical retrieval with reranking."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for, as a question or phrase.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"Passages to return (1-{MAX_RESULTS}). Default 5.",
                        "minimum": 1,
                        "maximum": MAX_RESULTS,
                    },
                    "mode": {
                        "type": "string",
                        "enum": [m.value for m in SearchMode],
                        "description": (
                            "fast: one vector search, no LLM calls. balanced: "
                            "query expansion plus reranking. deep: wider pool. "
                            "Default balanced."
                        ),
                    },
                    "corpus": {
                        "type": "string",
                        "description": "Which corpus to search. Omit for the default.",
                    },
                    "year": {
                        "type": "string",
                        "description": "Restrict to documents of this publication year.",
                    },
                    "paper_id": {
                        "type": "string",
                        "description": "Restrict to one document.",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "answer_question",
            "description": (
                "Answer a question from the corpus, with inline citations and a "
                "grounding check. Spends an LLM call. Prefer search_corpus "
                "unless you specifically want this system's own verified "
                "answer rather than the underlying passages."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": [m.value for m in SearchMode],
                        "description": "Default balanced. deep adds a grounding check.",
                    },
                    "corpus": {"type": "string"},
                },
                "required": ["question"],
            },
        },
        {
            "name": "list_corpora",
            "description": (
                "List the available corpora with what each contains, so you "
                "can tell whether a question is answerable before searching."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "corpus_info",
            "description": (
                "Describe one corpus: what kind of documents it holds, how "
                "many, its subject vocabulary and whether it is indexed."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "corpus": {"type": "string", "description": "Omit for the default."}
                },
            },
        },
    ]


class MCPServer:
    """JSON-RPC 2.0 over stdio, serving Vectora's retrieval as MCP tools."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._containers: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._initialised = False

    # ── container management ──────────────────────────────────────────────

    async def _container(self, corpus: str | None) -> Any:
        """Get or build the container for a corpus.

        Cached per corpus: a client may search several in one session and
        reloading the models each time would make the second call as slow as
        the first. The lock prevents two concurrent calls both paying for
        the build.
        """
        key = corpus or self.settings.corpus
        async with self._lock:
            if key not in self._containers:
                from vectora.api.deps import build_container

                log.info("building container for corpus", extra={"corpus": key})
                self._containers[key] = await asyncio.to_thread(
                    build_container, self.settings, warm=True, corpus=corpus
                )
            return self._containers[key]

    async def aclose(self) -> None:
        for container in self._containers.values():
            await container.aclose()
        self._containers.clear()

    # ── protocol ─────────────────────────────────────────────────────────

    async def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC message. ``None`` for notifications."""
        method = message.get("method", "")
        msg_id = message.get("id")
        params = message.get("params") or {}

        # A notification has no id and must not be answered.
        if msg_id is None and method.startswith("notifications/"):
            if method == "notifications/initialized":
                self._initialised = True
            return None

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": __version__},
                    "instructions": (
                        "Vectora searches a corpus of documents with hybrid "
                        "retrieval. Call list_corpora first if you do not know "
                        "what is indexed. Prefer search_corpus and reason over "
                        "the passages yourself; answer_question spends an LLM "
                        "call to produce this system's own cited answer."
                    ),
                }
            elif method == "tools/list":
                result = {"tools": _tool_definitions()}
            elif method == "tools/call":
                result = await self._call_tool(
                    str(params.get("name", "")), params.get("arguments") or {}
                )
            elif method == "ping":
                result = {}
            else:
                return _error(msg_id, -32601, f"Unknown method: {method}")
        except Exception as exc:  # a tool failure must not kill the server
            log.exception("mcp request failed", extra={"method": method})
            return _error(msg_id, -32603, f"{type(exc).__name__}: {exc}")

        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    async def _call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool and wrap its output in MCP content blocks.

        Errors come back as ``isError`` content rather than a protocol error:
        a client should see "that corpus is not indexed" as a tool result it
        can act on, not as a transport failure.
        """
        handlers = {
            "search_corpus": self._search,
            "answer_question": self._answer,
            "list_corpora": self._list_corpora,
            "corpus_info": self._corpus_info,
        }
        handler = handlers.get(name)
        if handler is None:
            return _tool_error(f"Unknown tool: {name}")
        try:
            payload = await handler(args)
        except Exception as exc:
            log.warning("tool failed", extra={"tool": name, "err": str(exc)[:160]})
            return _tool_error(f"{type(exc).__name__}: {exc}")
        return {
            "content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}],
            "isError": False,
        }

    # ── tools ─────────────────────────────────────────────────────────────

    async def _search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")

        container = await self._container(args.get("corpus"))
        if not container.store.exists():
            raise RuntimeError(
                f"Corpus '{container.corpus.name}' is not indexed. "
                f"Run `vectora index` for it first."
            )

        mode = _parse_mode(args.get("mode"))
        top_k = max(1, min(int(args.get("top_k") or 5), MAX_RESULTS))
        filters = {
            k: v
            for k, v in (("year", args.get("year")), ("paper_id", args.get("paper_id")))
            if v
        }

        from vectora.generation.router import classify

        query_type, _ = await classify(
            query, llm=container.llm, settings=container.settings
        )
        with request_context():
            chunks, diag = await container.engine.retriever.retrieve(
                query,
                mode=mode,
                query_type=query_type,
                top_k=top_k,
                filters=filters or None,
            )

        return {
            "corpus": container.corpus.name,
            "query": query,
            "query_type": query_type.value,
            "result_count": len(chunks),
            "results": [
                {
                    "rank": i,
                    "score": round(rc.score, 4),
                    "citation": (
                        f"{rc.chunk.title} ({rc.chunk.year or 'n.d.'})"
                        f"{', ' + rc.chunk.section if rc.chunk.section else ''}"
                        f"{', p.' + str(rc.chunk.page) if rc.chunk.page else ''}"
                    ),
                    "document": rc.chunk.paper_id,
                    "title": rc.chunk.title,
                    "authors": rc.chunk.authors,
                    "year": rc.chunk.year,
                    "doi": rc.chunk.doi,
                    "section": rc.chunk.section,
                    "page": rc.chunk.page,
                    "text": rc.chunk.content[:MAX_CONTENT_CHARS],
                    "truncated": len(rc.chunk.content) > MAX_CONTENT_CHARS,
                }
                for i, rc in enumerate(chunks, start=1)
            ],
            "retrieval": {
                "mode": mode.value,
                "dense_hits": diag.dense_hits,
                "sparse_hits": diag.sparse_hits,
                "reranked": diag.reranked,
                "degraded": diag.degraded,
            },
            "note": (
                "Nothing matched well enough to be useful; this corpus "
                "probably does not cover the question."
                if not chunks
                else "Cite these by title and page when you use them."
            ),
        }

    async def _answer(self, args: dict[str, Any]) -> dict[str, Any]:
        question = str(args.get("question") or "").strip()
        if not question:
            raise ValueError("question is required")

        container = await self._container(args.get("corpus"))
        if container.llm is None:
            raise RuntimeError(
                "No OPENAI_API_KEY configured, so this server cannot generate "
                "answers. Use search_corpus instead — it needs no LLM."
            )
        if not container.store.exists():
            raise RuntimeError(f"Corpus '{container.corpus.name}' is not indexed.")

        with request_context():
            answer = await container.engine.answer(
                question, mode=_parse_mode(args.get("mode"))
            )

        return {
            "corpus": container.corpus.name,
            "question": question,
            "answer": answer.answer,
            "refused": answer.fallback is not None,
            "refusal_reason": answer.fallback.reason if answer.fallback else None,
            "suggestions": answer.fallback.suggestions if answer.fallback else [],
            "citations": [
                {
                    "marker": f"[{c.index}]",
                    "title": c.title,
                    "authors": c.authors,
                    "year": c.year,
                    "section": c.section,
                    "page": c.page,
                    "doi": c.doi,
                    "supporting_quote": c.quote,
                }
                for c in answer.citations
            ],
            "grounding": (
                {
                    "score": answer.grounding.score,
                    "verdict": answer.grounding.verdict,
                    "unsupported_claims": answer.grounding.unsupported_claims,
                }
                if answer.grounding
                else None
            ),
            "usage": {
                "llm_calls": answer.usage.calls,
                "tokens": answer.usage.total_tokens,
                "cost_usd": round(answer.usage.cost_usd, 6),
            },
        }

    async def _list_corpora(self, args: dict[str, Any]) -> dict[str, Any]:
        from vectora.corpus import CorpusRegistry

        registry = CorpusRegistry(self.settings.corpora_dir)
        profiles = registry.list()
        return {
            "default": self.settings.corpus,
            "count": len(profiles),
            "corpora": [
                {
                    "name": p.name,
                    "description": p.description,
                    "document_kind": p.document_kind,
                    "documents": p.documents,
                    "chunks": p.chunks,
                    "indexed_collection": p.collection,
                    "topics": p.domain_terms[:12],
                }
                for p in profiles
            ],
            "note": (
                "No corpora yet. Create one with `vectora corpus create`."
                if not profiles
                else "Pass a name as the `corpus` argument to search a specific one."
            ),
        }

    async def _corpus_info(self, args: dict[str, Any]) -> dict[str, Any]:
        container = await self._container(args.get("corpus"))
        profile = container.corpus.profile
        indexed = 0
        healthy = container.store.healthy()
        if healthy and container.store.exists():
            try:
                indexed = container.store.stats().vectors
            except Exception:
                healthy = False
        return {
            "name": profile.name,
            "description": profile.description,
            "document_kind": profile.document_kind,
            "documents": profile.documents,
            "chunks": profile.chunks,
            "vocabulary": profile.domain_terms,
            "source": profile.source,
            "indexed": bool(indexed),
            "indexed_vectors": indexed,
            "searchable": bool(indexed) and healthy,
            "can_answer": bool(indexed) and container.llm is not None,
            "notes": profile.notes,
        }


def _parse_mode(value: Any) -> SearchMode:
    try:
        return SearchMode(str(value)) if value else SearchMode.BALANCED
    except ValueError:
        return SearchMode.BALANCED


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _tool_error(message: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": message}], "isError": True}


async def serve_stdio(settings: Settings) -> None:
    """Read JSON-RPC frames from stdin, write responses to stdout.

    stdout is the protocol channel, so logging must already be on stderr
    before this is called — a stray print corrupts the stream and the client
    sees a parse error rather than anything diagnosable.
    """
    server = MCPServer(settings)
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin
    )

    log.info("mcp server ready", extra={"version": __version__})
    try:
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except json.JSONDecodeError as exc:
                sys.stdout.write(
                    json.dumps(_error(None, -32700, f"Parse error: {exc}")) + "\n"
                )
                sys.stdout.flush()
                continue

            response = await server.handle(message)
            if response is not None:
                sys.stdout.write(json.dumps(response, default=str) + "\n")
                sys.stdout.flush()
    finally:
        log.info("mcp server shutting down")
        await server.aclose()
