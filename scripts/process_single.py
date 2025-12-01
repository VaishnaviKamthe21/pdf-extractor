import argparse
import os

from dotenv import load_dotenv

from src.models.schemas import ProcessingConfig
from src.utils.pipeline_single import process_single_pdf


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Process a single chapter PDF")
    parser.add_argument("--pdf", required=True, help="Path to the chapter PDF")
    parser.add_argument("--board", default=os.getenv("DEFAULT_BOARD", "State Board Maharashtra"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--book", required=True)
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "en"))

    args = parser.parse_args()

    config = ProcessingConfig(
        board=args.board,
        subject=args.subject,
        grade=args.grade,
        book=args.book,
        language=args.language,
    )

    validated = process_single_pdf(args.pdf, config=config, output_dir="output")

    # Optional: print summary to console
    print(f"Lesson ID: {validated.lesson_id}")
    print(f"Chapter: {validated.chapter_no} - {validated.title}")
    print(f"Content length: {len(validated.content)} characters")


if __name__ == "__main__":
    main()
