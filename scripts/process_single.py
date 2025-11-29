import argparse
import logging
import os

from dotenv import load_dotenv

from src.models.schemas import ProcessingConfig
from src.utils.pipeline_single import process_single_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Process a single textbook PDF into JSON.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument("--board", default=None, help="Board name.")
    parser.add_argument("--subject", default=None, help="Subject name.")
    parser.add_argument("--grade", type=int, default=None, help="Grade (int).")
    parser.add_argument("--book", default=None, help="Book name.")
    parser.add_argument("--language", default=None, help="Language code, e.g. en/mr/hi.")
    return parser.parse_args()


def main():
    load_dotenv()

    args = parse_args()

    # Fallbacks from env if CLI args missing
    board = args.board or os.getenv("DEFAULT_BOARD", "State Board Maharashtra")
    subject = args.subject or os.getenv("DEFAULT_SUBJECT", "Science")
    grade = args.grade or int(os.getenv("DEFAULT_GRADE", "4"))
    book = args.book or os.getenv("DEFAULT_BOOK", "Play, Do, Learn")
    language = args.language or os.getenv("DEFAULT_LANGUAGE", "en")

    enable_ocr = os.getenv("ENABLE_OCR_FALLBACK", "true").lower() == "true"

    config = ProcessingConfig(
        board=board,
        subject=subject,
        grade=grade,
        book=book,
        language=language,
        enable_ocr_fallback=enable_ocr,
    )

    logger.info(f"Using config: {config}")

    validated = process_single_pdf(args.pdf, config=config, output_dir="output")

    logger.info(
        f"Done. lesson_id={validated.lesson_id}, "
        f"content_length={len(validated.content)}, "
        f"confidence={validated.confidence:.2f}"
    )


if __name__ == "__main__":
    main()
