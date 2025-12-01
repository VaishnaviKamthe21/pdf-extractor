from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from src.extractor.pymupdf_extractor import PyMuPDFExtractor
from src.models.schemas import ProcessingConfig, ExtractionResult, ValidatedResult

logger = logging.getLogger(__name__)


def process_single_pdf(
    pdf_path: str,
    config: ProcessingConfig,
    output_dir: str = "output",
) -> ValidatedResult:
    """
    End-to-end processing of a single chapter PDF (text-only):
      1) Derive chapter_no and title from filename
      2) Extract with PyMuPDF
      3) Merge pages into content
      4) Save JSON to output/
    """
    pdf_path = str(pdf_path)
    pdf_stem = Path(pdf_path).stem  # e.g. "Chapter_01_Where_the_mind_is_without_fear"
    logger.info(f"Processing chapter PDF: {pdf_path}")

    chapter_no, title = _parse_chapter_metadata_from_filename(pdf_stem)

    extractor = PyMuPDFExtractor(min_confidence=config.min_page_confidence)
    extraction: ExtractionResult = extractor.extract(
        pdf_path=pdf_path,
        board=config.board,
        subject=config.subject,
        grade=config.grade,
        book=config.book,
        language=config.language,
    )

    merged_text = _merge_pages_to_content(extraction)

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
        f"chapter={chapter_no}, title={title}, len(content)={len(validated.content)}"
    )

    return validated


def _parse_chapter_metadata_from_filename(stem: str) -> tuple[str, str]:
    """
    Parse chapter number and title from a filename like:
      'Chapter_01_Where_the_mind_is_without_fear'

    Returns (chapter_no, title):
      ('01', 'Where the mind is without fear')
    """
    parts = stem.split("_", 2)
    # Expected pattern: ["Chapter", "01", "Where_the_mind_is_without_fear"]
    if len(parts) >= 3 and parts[0].lower() == "chapter":
        chapter_no = parts[1]
        raw_title = parts[2]
    else:
        # Fallback: chapter_no = "1.0", title = stem with underscores replaced
        chapter_no = "1.0"
        raw_title = stem

    title = raw_title.replace("_", " ").strip()
    return chapter_no, title


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

    data = validated.model_dump()
    if isinstance(data.get("created_at"), datetime):
        data["created_at"] = data["created_at"].isoformat()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved validated JSON to: {out_path}")
