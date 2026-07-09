"""Pydantic v2 request/response models for the FastAPI service."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    query: str = Field(..., min_length=1)
    # CRAG pages; the service is self-contained/offline (no live web fetch).
    documents: list[str] | None = None
    query_time: str | None = None
    threshold_override: float | None = Field(None, ge=0.0, le=1.0)  # demo slider hook


class Citation(BaseModel):
    page_url: str
    page_name: str = ""


class AnswerResponse(BaseModel):
    answer: str | None                                   # fixed IDK string when abstained
    abstained: bool
    confidence: float = Field(..., ge=0.0, le=1.0)         # calibrated P(correct)
    reason: Literal["answered", "low_confidence", "false_premise"]
    citations: list[Citation] = []
    meta: dict = {}
