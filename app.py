"""
Research Paper RAG — App with Chat History
============================================
Manages chat sessions in SQLite. Each session stores the full
conversation history and passes it to OpenAI for context-aware responses.

Usage:
  python app.py
  Open: http://localhost:5000
"""

import os
import uuid
import json
import sqlite3
import logging
import textwrap
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, g, Response
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from generation import generate_with_history

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

DB_PATH = "chats.db"

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop("db", None)
    if db:
        db.close()

def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id        TEXT PRIMARY KEY,
                title     TEXT NOT NULL,
                created   TEXT NOT NULL,
                updated   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                citations  TEXT,
                created    TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)

_openai_client = None
_qdrant_client = None

def get_clients():
    global _openai_client, _qdrant_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Add it to your .env file.")
        _openai_client = OpenAI(api_key=api_key)
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url="http://localhost:6333")
    return _openai_client, _qdrant_client

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/sessions", methods=["GET"])
def list_sessions():
    db = get_db()
    rows = db.execute(
        "SELECT id, title, created, updated FROM sessions ORDER BY updated DESC"
    ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route("/sessions", methods=["POST"])
def create_session():
    db  = get_db()
    sid = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    db.execute(
        "INSERT INTO sessions (id, title, created, updated) VALUES (?, ?, ?, ?)",
        (sid, "New chat", now, now)
    )
    db.commit()
    return jsonify({"id": sid, "title": "New chat", "created": now, "updated": now})

@app.route("/sessions/<sid>", methods=["DELETE"])
def delete_session(sid):
    db = get_db()
    db.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
    db.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    db.commit()
    return jsonify({"ok": True})

@app.route("/sessions/<sid>/messages", methods=["GET"])
def get_messages(sid):
    db   = get_db()
    rows = db.execute(
        "SELECT role, content, citations, created FROM messages "
        "WHERE session_id = ? ORDER BY id ASC", (sid,)
    ).fetchall()
    msgs = []
    for r in rows:
        m = dict(r)
        m["citations"] = json.loads(m["citations"]) if m["citations"] else []
        msgs.append(m)
    return jsonify(msgs)

@app.route("/sessions/<sid>/chat", methods=["POST"])
def chat(sid):
    db    = get_db()
    data  = request.get_json()
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    session = db.execute("SELECT id, title FROM sessions WHERE id = ?", (sid,)).fetchone()
    if not session:
        return jsonify({"error": "Session not found"}), 404

    history_rows = db.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (sid,)
    ).fetchall()
    history = [{"role": r["role"], "content": r["content"]} for r in history_rows]

    filters = {}
    if data.get("year"):     filters["year"]     = data["year"]
    if data.get("paper_id"): filters["paper_id"] = data["paper_id"]

    try:
        oc, qc = get_clients()
        mode = data.get("mode", "balanced")
        # top_k is now driven by mode profile in retrieval.py
        # but we still pass it for route_config override support
        mode_top_k = {"fast": 8, "balanced": 15, "deep": 20}.get(mode, 15)
        result = generate_with_history(
            query         = query,
            history       = history,
            qdrant_client = qc,
            openai_client = oc,
            top_k         = mode_top_k,
            filters       = filters or None,
            mode          = mode,
        )
    except Exception as e:
        log.error(f"Generation error: {e}")
        return jsonify({"error": str(e)}), 500

    now = datetime.utcnow().isoformat()

    db.execute(
        "INSERT INTO messages (session_id, role, content, citations, created) VALUES (?,?,?,?,?)",
        (sid, "user", query, None, now)
    )

    citations_json = json.dumps([
        {"title": c.title, "authors": c.authors, "year": c.year,
         "doi": c.doi, "section": c.section, "page": c.page}
        for c in result.citations
    ])
    db.execute(
        "INSERT INTO messages (session_id, role, content, citations, created) VALUES (?,?,?,?,?)",
        (sid, "assistant", result.answer, citations_json, now)
    )

    if session["title"] == "New chat":
        title = query[:52] + "..." if len(query) > 52 else query
        db.execute("UPDATE sessions SET title = ?, updated = ? WHERE id = ?", (title, now, sid))
    else:
        db.execute("UPDATE sessions SET updated = ? WHERE id = ?", (now, sid))

    db.commit()

    # Serialize faithfulness result if present
    faithfulness_data = None
    if result.faithfulness:
        faithfulness_data = {
            "score":              result.faithfulness.score,
            "verdict":            result.faithfulness.verdict,
            "supported_claims":   result.faithfulness.supported_claims,
            "unsupported_claims": result.faithfulness.unsupported_claims,
            "summary":            result.faithfulness.summary,
        }

    # Serialize fallback if present
    fallback_data = None
    if result.fallback and result.fallback.triggered:
        fallback_data = {
            "triggered":      True,
            "reason":         result.fallback.reason,
            "message":        result.fallback.message,
            "suggestions":    result.fallback.suggestions or [],
            "related_topics": result.fallback.related_topics or [],
        }

    return jsonify({
        "answer":       result.answer,
        "citations":    json.loads(citations_json),
        "faithfulness": faithfulness_data,
        "query_type":   result.query_type or "unknown",
        "fallback":     fallback_data,
    })

# ─────────────────────────────────────────────
# PDF Export
# ─────────────────────────────────────────────

def build_export_html(session: dict, messages: list) -> str:
    """
    Build a self-contained HTML document for the conversation.
    weasyprint renders this to PDF.
    """
    title    = session["title"]
    created  = session["created"][:10]

    def fmt_citations(citations):
        if not citations:
            return ""
        items = ""
        seen  = set()
        for c in citations:
            key = c.get("title", "")
            if key in seen:
                continue
            seen.add(key)
            author  = c.get("authors","").split(",")[0].strip() or "Unknown"
            year    = c.get("year") or "n.d."
            sec     = c.get("section","")
            page    = c.get("page","")
            doi     = c.get("doi","")
            doi_str = f'<a href="https://doi.org/{doi}">{doi}</a>' if doi else ""
            items  += f"""<div class="cite-item">
                <span class="cite-title">{c.get("title","Untitled")}</span>
                &nbsp;·&nbsp;{author} et al. ({year})
                &nbsp;·&nbsp;{sec} p.{page}
                {"&nbsp;·&nbsp;" + doi_str if doi_str else ""}
            </div>"""
        return f'<div class="citations"><div class="cite-label">References</div>{items}</div>'

    rows = ""
    for m in messages:
        role     = m["role"]
        body     = m["content"].replace("<","&lt;").replace(">","&gt;")
        # Convert newlines to paragraphs
        paras    = "".join(f"<p>{p}</p>" for p in body.split("\n\n") if p.strip())
        cites    = fmt_citations(m.get("citations") or [])
        ts       = m.get("created","")[:16].replace("T"," ")
        if role == "user":
            rows += f"""<div class="msg user">
                <div class="msg-meta">You &nbsp;·&nbsp; {ts}</div>
                <div class="msg-body user-body">{paras}</div>
            </div>"""
        else:
            rows += f"""<div class="msg assistant">
                <div class="msg-meta">Vectora &nbsp;·&nbsp; {ts}</div>
                <div class="msg-body">{paras}{cites}</div>
            </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400&display=swap');

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'DM Mono', monospace;
    font-size: 11pt;
    font-weight: 300;
    line-height: 1.7;
    color: #1a1a18;
    background: #fff;
    padding: 0;
  }}
  .cover {{
    padding: 60px 60px 40px;
    border-bottom: 2px solid #1a1a18;
    margin-bottom: 40px;
  }}
  .cover-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 28pt;
    color: #1a1a18;
    line-height: 1.2;
    margin-bottom: 8px;
  }}
  .cover-meta {{
    font-size: 9pt;
    color: #888;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}
  .cover-logo {{
    font-family: 'DM Serif Display', serif;
    font-size: 13pt;
    color: #c8b99a;
    margin-bottom: 20px;
  }}
  .messages {{
    padding: 0 60px 60px;
    display: flex;
    flex-direction: column;
    gap: 32px;
  }}
  .msg {{ page-break-inside: avoid; }}
  .msg-meta {{
    font-size: 8.5pt;
    color: #aaa;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 6px;
  }}
  .msg-body {{
    border: 1px solid #e0ddd6;
    border-radius: 6px;
    padding: 14px 18px;
    background: #faf9f7;
  }}
  .user-body {{
    background: #f5f2ec;
    border-color: #d4ccbc;
    color: #5a4e38;
  }}
  .msg-body p {{ margin-bottom: 8px; }}
  .msg-body p:last-child {{ margin-bottom: 0; }}
  .citations {{
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid #e0ddd6;
  }}
  .cite-label {{
    font-size: 8pt;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 6px;
  }}
  .cite-item {{
    font-size: 9pt;
    color: #888;
    padding: 3px 0 3px 10px;
    border-left: 2px solid #d4ccbc;
    margin-bottom: 4px;
    line-height: 1.5;
  }}
  .cite-title {{ color: #8a7056; }}
  a {{ color: #8a7056; }}
  @page {{
    margin: 0;
    size: A4;
  }}
</style>
</head>
<body>
  <div class="cover">
    <div class="cover-logo">Vectora</div>
    <div class="cover-title">{title}</div>
    <div class="cover-meta">Exported {created} &nbsp;·&nbsp; {len(messages)} messages</div>
  </div>
  <div class="messages">
    {rows}
  </div>
</body>
</html>"""


@app.route("/sessions/<sid>/export/pdf", methods=["GET"])
def export_pdf(sid):
    db = get_db()

    session = db.execute(
        "SELECT id, title, created FROM sessions WHERE id = ?", (sid,)
    ).fetchone()
    if not session:
        return jsonify({"error": "Session not found"}), 404

    rows = db.execute(
        "SELECT role, content, citations, created FROM messages "
        "WHERE session_id = ? ORDER BY id ASC", (sid,)
    ).fetchall()
    messages = []
    for r in rows:
        m = dict(r)
        m["citations"] = json.loads(m["citations"]) if m["citations"] else []
        messages.append(m)

    if not messages:
        return jsonify({"error": "No messages to export"}), 400

    html = build_export_html(dict(session), messages)

    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html).write_pdf()
    except ImportError:
        return jsonify({
            "error": "weasyprint not installed. Run: pip install weasyprint"
        }), 500
    except Exception as e:
        log.error(f"PDF generation error: {e}")
        return jsonify({"error": str(e)}), 500

    safe_title = "".join(c if c.isalnum() or c in "- _" else "_" for c in session["title"])[:50]
    filename   = f"vectora_{safe_title}_{session['created'][:10]}.pdf"

    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


if __name__ == "__main__":
    init_db()
    print("\n  RAG Chatbot running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
