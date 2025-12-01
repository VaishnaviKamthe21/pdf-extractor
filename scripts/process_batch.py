"""How to run it:

Put all chapter PDFs (named like Chapter_01_..., Chapter_02_..., etc.) in a folder, e.g. data/chapters.

Then from project root:

python -m scripts.process_batch --input-dir data/chapters --subject English --grade 10 --book "English Balbharti" --workers 4
"""
import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from src.models.schemas import ProcessingConfig
from src.utils.pipeline_single import process_single_pdf


def _process_one(pdf_path: Path, config: ProcessingConfig, output_dir: str) -> tuple[str, str]:
    """Helper to process a single PDF and return (pdf_name, lesson_id)."""
    res = process_single_pdf(str(pdf_path), config=config, output_dir=output_dir)
    return pdf_path.name, res.lesson_id


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Process multiple chapter PDFs in a folder")
    parser.add_argument("--input-dir", required=True, help="Directory containing chapter PDFs")
    parser.add_argument("--board", default=os.getenv("DEFAULT_BOARD", "State Board Maharashtra"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--grade", type=int, required=True)
    parser.add_argument("--book", required=True)
    parser.add_argument("--language", default=os.getenv("DEFAULT_LANGUAGE", "en"))
    parser.add_argument("--output-dir", default="output", help="Where to store JSON outputs")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = args.output_dir

    if not input_dir.is_dir():
        raise ValueError(f"Input dir does not exist or is not a directory: {input_dir}")

    # Collect all PDFs
    pdf_files = sorted(p for p in input_dir.glob("*.pdf") if p.is_file())
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files in {input_dir}")

    config = ProcessingConfig(
        board=args.board,
        subject=args.subject,
        grade=args.grade,
        book=args.book,
        language=args.language,
    )

    results = []

    # Parallel processing
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pdf = {
            executor.submit(_process_one, pdf_path, config, output_dir): pdf_path
            for pdf_path in pdf_files
        }

        for future in as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                pdf_name, lesson_id = future.result()
                results.append((pdf_name, lesson_id))
                print(f"[OK] {pdf_name} -> lesson_id={lesson_id}")
            except Exception as e:
                print(f"[ERROR] {pdf_path.name}: {e}")

    print("\nBatch processing complete.")
    print(f"Total processed: {len(results)} / {len(pdf_files)}")


if __name__ == "__main__":
    main()
