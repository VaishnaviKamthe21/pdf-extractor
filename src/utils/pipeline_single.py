from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.extractor.pymupdf_extractor import PyMuPDFExtractor
from src.extractor.ocr_tesseract import TesseractOCR
from src.models.schemas import ProcessingConfig, ExtractionResult, ValidatedResult

logger = logging.getLogger(__name__)


def process_single_pdf(
    pdf_path: str,
    config: ProcessingConfig,
    output_dir: str = "output",
) -> ValidatedResult:
    """
    End-to-end processing of a single PDF:
      1) Extract with PyMuPDF
      2) Optional OCR fallback on low-confidence pages
      3) Merge content into a ValidatedResult
      4) Save JSON to output/
    """
    pdf_path = str(pdf_path)
    logger.info(f"Processing PDF: {pdf_path}")

    extractor = PyMuPDFExtractor(min_confidence=config.min_page_confidence)
    extraction: ExtractionResult = extractor.extract(
        pdf_path=pdf_path,
        board=config.board,
        subject=config.subject,
        grade=config.grade,
        book=config.book,
        language=config.language,
    )

    # # Optional OCR fallback
    # if config.enable_ocr_fallback:
    #     ocr = TesseractOCR(dpi=config.ocr_dpi, lang=config.ocr_lang)
    #     extraction = ocr.apply_ocr_to_results(pdf_path, extraction)

    # Merge all page texts into one content string
    merged_text = _merge_pages_to_content(extraction)

    # For now, simple placeholders for chapter_no and title
    chapter_no = "1.0"
    title = "Auto Extracted Lesson"

    # Compute overall confidence and counts
    overall_conf = min(p.confidence for p in extraction.pages) if extraction.pages else 0.0
    image_count = sum(p.image_count for p in extraction.pages)
    table_count = sum(p.table_count for p in extraction.pages)

    validated = ValidatedResult(
        lesson_id=extraction.lesson_id,
        board=config.board,
        subject=config.subject,
        grade=config.grade,
        book=config.book,
        chapter_no=chapter_no,
        title=title,
        content=merged_text,
        language=config.language,
        page_number=None,
        confidence=overall_conf,
        image_count=image_count,
        table_count=table_count,
        created_at=datetime.utcnow(),
    )

    _save_validated_json(validated, output_dir)

    logger.info(
        f"Finished processing {pdf_path} -> lesson_id={validated.lesson_id}, "
        f"len(content)={len(validated.content)}"
    )

    return validated


def _merge_pages_to_content(extraction: ExtractionResult) -> str:
    """Simple merge: join page texts with page separators."""
    parts = []
    for page in extraction.pages:
        header = f"\n\n=== Page {page.page_number} ===\n\n"
        parts.append(header + (page.raw_text or "").strip())
    return "\n".join(parts).strip()


def _save_validated_json(validated: ValidatedResult, output_dir: str) -> None:
    """Save the ValidatedResult to a JSON file in output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    filename = f"{validated.lesson_id}_validated_{ts}.json"
    out_path = Path(output_dir) / filename

    # Use model_dump + convert datetimes to ISO strings
    data = validated.model_dump()
    if isinstance(data.get("created_at"), datetime):
        data["created_at"] = data["created_at"].isoformat()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved validated JSON to: {out_path}")
