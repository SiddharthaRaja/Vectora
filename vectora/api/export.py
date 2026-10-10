"""
Conversation export.

The styling is kept from the original implementation — it was good — with
the HTML escaping fixed. The old version escaped ``<`` and ``>`` but not
``&``, so ``&amp;`` in a paper title rendered as ``&`` and an entity in the
source text could still break the markup; and citation fields were
interpolated raw, so a DOI or title containing a quote could break out of
its attribute.
"""

from __future__ import annotations

import html
from collections.abc import Sequence

from vectora.models import Citation, Message, Session, first_author

_CSS = """
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@300;400&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'DM Mono', monospace; font-size: 10.5pt; font-weight: 300;
         line-height: 1.7; color: #1a1a18; background: #fff; }
  .cover { padding: 56px 56px 32px; border-bottom: 2px solid #1a1a18; margin-bottom: 36px; }
  .cover-logo { font-family: 'DM Serif Display', serif; font-size: 13pt; color: #8a7056;
                margin-bottom: 18px; letter-spacing: .02em; }
  .cover-title { font-family: 'DM Serif Display', serif; font-size: 26pt; line-height: 1.18;
                 margin-bottom: 10px; }
  .cover-meta { font-size: 8.5pt; color: #8d8d86; letter-spacing: .1em; text-transform: uppercase; }
  .messages { padding: 0 56px 56px; display: flex; flex-direction: column; gap: 28px; }
  .msg { page-break-inside: avoid; }
  .msg-meta { font-size: 8pt; color: #a8a8a0; letter-spacing: .1em; text-transform: uppercase;
              margin-bottom: 6px; }
  .msg-body { border: 1px solid #e0ddd6; border-radius: 6px; padding: 14px 18px; background: #faf9f7; }
  .user-body { background: #f5f2ec; border-color: #d4ccbc; color: #5a4e38; }
  .msg-body p { margin-bottom: 8px; }
  .msg-body p:last-child { margin-bottom: 0; }
  .grounding { margin-top: 12px; font-size: 8.5pt; color: #8a7056;
               letter-spacing: .06em; text-transform: uppercase; }
  .citations { margin-top: 14px; padding-top: 12px; border-top: 1px solid #e0ddd6; }
  .cite-label { font-size: 7.5pt; letter-spacing: .12em; text-transform: uppercase;
                color: #a8a8a0; margin-bottom: 6px; }
  .cite-item { font-size: 8.5pt; color: #7d7d76; padding: 4px 0 4px 10px;
               border-left: 2px solid #d4ccbc; margin-bottom: 4px; line-height: 1.55; }
  .cite-title { color: #8a7056; }
  .cite-quote { color: #9a9a92; font-style: italic; display: block; margin-top: 2px; }
  a { color: #8a7056; text-decoration: none; }
  @page { margin: 0; size: A4; }
"""


def _esc(value: object) -> str:
    """Escape for HTML text and attributes, ``&`` included."""
    return html.escape(str(value or ""), quote=True)


def _paragraphs(text: str) -> str:
    blocks = [b.strip() for b in (text or "").split("\n\n") if b.strip()]
    if not blocks:
        return "<p></p>"
    return "".join(f"<p>{_esc(b).replace(chr(10), '<br>')}</p>" for b in blocks)


def _citations_block(citations: Sequence[Citation]) -> str:
    if not citations:
        return ""
    items: list[str] = []
    seen: set[str] = set()
    for c in citations:
        key = c.paper_id or c.title
        if key in seen:
            continue
        seen.add(key)
        who = _esc(first_author(c.authors) or "Unknown")
        year = _esc(c.year or "n.d.")
        where = ", ".join(
            x for x in (_esc(c.section), f"p.{c.page}" if c.page else "") if x
        )
        doi = (
            f' · <a href="https://doi.org/{_esc(c.doi)}">{_esc(c.doi)}</a>'
            if c.doi
            else ""
        )
        quote = (
            f'<span class="cite-quote">“{_esc(c.quote)}”</span>' if c.quote else ""
        )
        items.append(
            f'<div class="cite-item">'
            f'<span class="cite-title">{_esc(c.title) or "Untitled"}</span>'
            f" · {who} et al. ({year})"
            f"{' · ' + where if where else ''}{doi}{quote}"
            f"</div>"
        )
    return f'<div class="citations"><div class="cite-label">References</div>{"".join(items)}</div>'


def build_conversation_html(session: Session, messages: Sequence[Message]) -> str:
    """Self-contained HTML document for one conversation."""
    rows: list[str] = []
    for m in messages:
        stamp = _esc(m.created_at.strftime("%Y-%m-%d %H:%M"))
        if m.role == "user":
            rows.append(
                f'<div class="msg"><div class="msg-meta">You · {stamp}</div>'
                f'<div class="msg-body user-body">{_paragraphs(m.content)}</div></div>'
            )
            continue

        grounding = ""
        meta = m.metadata or {}
        if isinstance(meta.get("grounding"), dict) and meta["grounding"].get("score") is not None:
            g = meta["grounding"]
            grounding = (
                f'<div class="grounding">{_esc(g.get("score"))}% grounded'
                f'{" · " + _esc(g.get("summary")) if g.get("summary") else ""}</div>'
            )
        rows.append(
            f'<div class="msg"><div class="msg-meta">Vectora · {stamp}</div>'
            f'<div class="msg-body">{_paragraphs(m.content)}'
            f"{grounding}{_citations_block(m.citations)}</div></div>"
        )

    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"UTF-8\">"
        f"<title>{_esc(session.title)}</title><style>{_CSS}</style></head><body>"
        f'<div class="cover"><div class="cover-logo">Vectora</div>'
        f'<div class="cover-title">{_esc(session.title)}</div>'
        f'<div class="cover-meta">Exported {session.created_at:%d %B %Y}'
        f" · {len(messages)} messages</div></div>"
        f'<div class="messages">{"".join(rows)}</div></body></html>'
    )


def render_pdf(html_text: str) -> bytes:
    """Render HTML to PDF. Raises ``ImportError`` if weasyprint is absent."""
    from weasyprint import HTML

    return HTML(string=html_text).write_pdf()
