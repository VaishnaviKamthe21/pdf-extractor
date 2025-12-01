from __future__ import annotations

import logging
from typing import List

import fitz  # PyMuPDF

from src.models.schemas import PageBlock, PageResult, ExtractionResult

logger = logging.getLogger(__name__)


class PyMuPDFExtractor:
    """Extract text, blocks, and basic metadata from a PDF using PyMuPDF."""

    def __init__(self, min_confidence: float = 0.85):
        self.min_confidence = min_confidence

    def extract(self, pdf_path: str,
                board: str | None = None,
                subject: str | None = None,
                grade: int | None = None,
                book: str | None = None,
                language: str | None = None) -> ExtractionResult:
        """Main entry: extract all pages from a PDF into an ExtractionResult."""
        logger.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(pdf_path)

        pages: List[PageResult] = []
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1

            # Extract full text
            raw_text = page.get_text("text") or ""

            # Extract structured blocks (dict mode gives positions)
            blocks_data = page.get_text("dict")["blocks"]
            blocks: List[PageBlock] = []
            for b in blocks_data:
                if "lines" not in b:
                    continue
                # Join all spans in the block
                text_parts = []
                for line in b["lines"]:
                    for span in line["spans"]:
                        text_parts.append(span.get("text", ""))
                text = " ".join(t.strip() for t in text_parts if t.strip())
                if not text:
                    continue
                x0, y0, x1, y1 = b["bbox"]
                blocks.append(PageBlock(text=text, x0=x0, y0=y0, x1=x1, y1=y1))

            # Count images on page
            image_count = len(page.get_images())

            # Very simple confidence heuristic
            confidence = self._estimate_confidence(raw_text)

            page_result = PageResult(
                page_number=page_number,
                raw_text=raw_text,
                blocks=blocks,
                image_count=image_count,
                table_count=0,  
                confidence=confidence,
                needs_ocr=confidence < self.min_confidence,
            )
            pages.append(page_result)

        extraction = ExtractionResult(
            pdf_path=pdf_path,
            pages=pages,
            board=board,
            subject=subject,
            grade=grade,
            book=book,
            language=language,
        )

        logger.info(
            f"Extracted {len(pages)} pages from {pdf_path} "
            f"(min_confidence={self.min_confidence})."
        )


        return extraction

    def _estimate_confidence(self, text: str) -> float:
        """Naive confidence estimator: low if text is empty or mostly weird chars."""
        if not text or not text.strip():
            return 0.0

        total = len(text)
        # Count characters that look like normal letters / digits / punctuation
        good = sum(ch.isalnum() or ch.isspace() or ch in ".,;:-_()[]{}!?\"'" for ch in text)
        ratio = good / total

        # If ratio is high, confidence high
        if ratio > 0.9:
            return 1.0
        if ratio > 0.7:
            return 0.9
        if ratio > 0.5:
            return 0.7
        return 0.4
