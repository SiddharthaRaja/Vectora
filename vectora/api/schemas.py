"""
Request and response schemas for the HTTP API.

The response shapes the shipped frontend already consumes are preserved
exactly — ``answer``, ``citations``, ``faithfulness``, ``fallback``,
``query_type`` — and everything new (``diagnostics``, ``usage``,
``sources``) is additive. That is why the existing ``static/index.html``
keeps working unchanged against this rewrite.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vectora.config import SearchMode


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str = Field(min_length=1, max_length=4000)
    mode: SearchMode = SearchMode.BALANCED
    year: str | None = None
    paper_id: str | None = None
    section: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        text = v.strip()
        if not text:
            raise ValueError("query cannot be blank")
        return text

    @field_validator("year")
    @classmethod
    def _year(cls, v: str | None) -> str | None:
        if v in (None, ""):
            return None
        text = str(v).strip()
        if not (text.isdigit() and len(text) == 4):
            raise ValueError("year must be four digits")
        return text

    def filters(self) -> dict[str, Any] | None:
        out = {
            k: v
            for k, v in (
                ("year", self.year),
                ("paper_id", self.paper_id),
                ("section", self.section),
            )
            if v
        }
        return out or None


class SessionOut(BaseModel):
    id: str
    title: str
    created: str
    updated: str
    message_count: int = 0


class CitationOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = 0
    title: str = ""
    authors: str = ""
    year: str = ""
    doi: str = ""
    section: str = ""
    page: int = 0
    quote: str = ""


class MessageOut(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[CitationOut] = Field(default_factory=list)
    created: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class HealthOut(BaseModel):
    status: str
    version: str
    env: str


class ErrorOut(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None
