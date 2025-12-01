import argparse
import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Test query against Pinecone index")
    parser.add_argument("--query", required=True, help="User question or search text")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--namespace", default=None, help="Namespace used during indexing")
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key or not index_name:
        raise RuntimeError("PINECONE_API_KEY or PINECONE_INDEX_NAME not set in .env")

    # 1) Connect to Pinecone
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # 2) Create same embedding model used for indexing
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 3) Embed the query text
    query_vec = embeddings.embed_query(args.query)

    # 4) Query Pinecone
    response = index.query(
        vector=query_vec,
        top_k=args.top_k,
        namespace=args.namespace,
        include_metadata=True,
    )

    # 5) Print results
    print(f"\nTop {args.top_k} results for query: {args.query}\n")
    for match in response.get("matches", []):
        score = match.get("score")
        meta = match.get("metadata", {})
        chunk_text = meta.get("chunk_text", "")
        lesson_id = meta.get("lesson_id")
        chapter_no = meta.get("chapter_no")
        title = meta.get("title")

        print("------------------------------------------------------------")
        print(f"Score: {score:.4f}")
        print(f"Lesson: {lesson_id}")
        print(f"Chapter: {chapter_no} - {title}")
        print(f"Text: {chunk_text[:400]}...")
    print("\nDone.")

if __name__ == "__main__":
    main()
