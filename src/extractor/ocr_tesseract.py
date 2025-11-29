from __future__ import annotations

import logging
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from src.models.schemas import PageResult, OCRResult, ExtractionResult

logger = logging.getLogger(__name__)

import pytesseract

# tesseract is not on PATH:
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


class TesseractOCR:
    """Run Tesseract OCR on low-confidence pages."""

    def __init__(self, dpi: int = 300, lang: str = "mar+hin+eng"):
        self.dpi = dpi
        self.lang = lang

    def ocr_page(self, page: fitz.Page, page_number: int) -> OCRResult:
        """Render a single page as an image and run OCR."""
        logger.info(f"Running OCR on page {page_number} with lang={self.lang}")

        pix = page.get_pixmap(dpi=self.dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(
            img,
            lang=self.lang,
            config="--psm 6 -c preserve_interword_spaces=1",
        )

        confidence = self._estimate_ocr_confidence(text)

        return OCRResult(
            page_number=page_number,
            text=text,
            language=self.lang,
            confidence=confidence,
        )

    def apply_ocr_to_results(self, pdf_path: str, extraction: ExtractionResult) -> ExtractionResult:
        """
        For each page with needs_ocr=True, run Tesseract and update
        raw_text + confidence.
        """
        doc = fitz.open(pdf_path)
        updated_pages: List[PageResult] = []

        for page_result in extraction.pages:
            if not page_result.needs_ocr:
                updated_pages.append(page_result)
                continue

            page = doc[page_result.page_number - 1]
            ocr_res = self.ocr_page(page, page_result.page_number)

            updated_pages.append(
                PageResult(
                    page_number=page_result.page_number,
                    raw_text=ocr_res.text,
                    blocks=page_result.blocks,  # we keep old blocks for now
                    image_count=page_result.image_count,
                    table_count=page_result.table_count,
                    confidence=ocr_res.confidence,
                    needs_ocr=False,
                )
            )

        extraction.pages = updated_pages
        return extraction

    def _estimate_ocr_confidence(self, text: str) -> float:
        """Very simple heuristic: empty or very short => low confidence."""
        if not text or not text.strip():
            return 0.0
        length = len(text.strip())
        if length < 20:
            return 0.4
        if length < 100:
            return 0.7
        return 0.9
