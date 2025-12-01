from __future__ import annotations

import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone

from src.models.schemas import ValidatedResult

logger = logging.getLogger(__name__)


class PineconeVectorizer:
    """
    Chunk lesson content, embed with local Hugging Face model,
    and upsert into Pinecone.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: str = "intfloat/multilingual-e5-large",
    ):
        # Connect to Pinecone index
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)

        # Recursive text splitter for chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Local embedding model from Hugging Face
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

    def _embed(self, text: str) -> List[float]:
        """Embed a single chunk of text."""
        return self.embeddings.embed_query(text)

    def upsert_validated_results(
        self,
        results: List[ValidatedResult],
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> None:
        """Chunk, embed, and upsert a list of ValidatedResult objects."""
        for res in results:
            self._upsert_single(res, namespace=namespace, batch_size=batch_size)

    def _upsert_single(
        self,
        res: ValidatedResult,
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> None:
        """Process one lesson/chapter and upsert its chunks."""
        text = res.content or ""
        if not text.strip():
            logger.warning(f"Empty content for lesson_id={res.lesson_id}, skipping upsert.")
            return

        # 1) Chunk with RecursiveCharacterTextSplitter
        chunks = self.splitter.split_text(text)
        logger.info(f"Lesson {res.lesson_id}: split into {len(chunks)} chunks.")

        # 2) Embed each chunk and prepare vectors
        vectors = []
        for i, chunk in enumerate(chunks):
            emb = self._embed(chunk)
            meta = {
                "lesson_id": res.lesson_id,
                "board": res.board,
                "subject": res.subject,
                "grade": res.grade,
                "book": res.book,
                "chapter_no": res.chapter_no,
                "title": res.title,
                "language": res.language,
                "chunk_id": i,
                "chunk_text": chunk[:1000],
            }
            vectors.append(
                {
                    "id": f"{res.lesson_id}_{i}",
                    "values": emb,
                    "metadata": meta,
                }
            )

            # Upsert in batches
            if len(vectors) >= batch_size:
                self._upsert_batch(vectors, namespace)
                vectors = []

        if vectors:
            self._upsert_batch(vectors, namespace)

    def _upsert_batch(self, vectors, namespace: str | None) -> None:
        """Helper to upsert one batch of vectors into Pinecone."""
        self.index.upsert(vectors=vectors, namespace=namespace)
        logger.info(f"Upserted batch of {len(vectors)} vectors to Pinecone (ns={namespace}).")
