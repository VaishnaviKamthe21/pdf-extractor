from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class PageBlock(BaseModel):
    """Single text block on a page (from PyMuPDF)."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float


class PageResult(BaseModel):
    """Result of processing a single page."""

    page_number: int
    raw_text: str
    blocks: List[PageBlock] = Field(default_factory=list)
    image_count: int = 0  
    table_count: int = 0
    confidence: float = 1.0


class ExtractionResult(BaseModel):
    """Full extraction result for one chapter PDF."""

    pdf_path: str
    lesson_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pages: List[PageResult]
    board: Optional[str] = None
    subject: Optional[str] = None
    grade: Optional[int] = None
    book: Optional[str] = None
    chapter_no: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ValidatedResult(BaseModel):
    """
    Final structured object matching your target JSON format
    for one chapter/lesson.
    """

    lesson_id: str
    board: str
    subject: str
    grade: int
    book: str
    chapter_no: str
    title: str
    content: str
    language: str
    page_number: Optional[int] = None
    confidence: float = 1.0
    image_count: int = 0
    table_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ProcessingConfig(BaseModel):
    """
    Configuration for processing a single chapter PDF.
    """

    board: str
    subject: str
    grade: int
    book: str
    language: str = "en"
    min_page_confidence: float = 0.85
    pinecone_index_name: str = "textbooks-prod"
    chunk_size: int = 512
    chunk_overlap: int = 50
