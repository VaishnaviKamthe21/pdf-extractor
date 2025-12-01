import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.models.schemas import ValidatedResult
from src.vectorizer.pinecone_vectorizer import PineconeVectorizer


def load_validated_results(output_dir: str) -> list[ValidatedResult]:
    """Load all validated chapter JSONs from output_dir."""
    results: list[ValidatedResult] = []
    for path in sorted(Path(output_dir).glob("*_validated_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results.append(ValidatedResult(**data))
    return results


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Index chapter JSONs into Pinecone")
    parser.add_argument("--output-dir", default="output", help="Folder with *_validated_*.json")
    parser.add_argument("--namespace", default=None, help="Optional Pinecone namespace")
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        raise RuntimeError("PINECONE_API_KEY or PINECONE_INDEX_NAME not set in .env")

    vectorizer = PineconeVectorizer(
        api_key=api_key,
        index_name=index_name,
        chunk_size=512,
        chunk_overlap=50,
        model_name="intfloat/multilingual-e5-large",
    )

    results = load_validated_results(args.output_dir)
    if not results:
        print(f"No validated JSON files found in {args.output_dir}")
        return

    print(f"Indexing {len(results)} chapters into Pinecone index '{index_name}'...")
    vectorizer.upsert_validated_results(results, namespace=args.namespace)
    print("Indexing complete.")


if __name__ == "__main__":
    main()
