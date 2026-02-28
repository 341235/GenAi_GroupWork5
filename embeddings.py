"""
embeddings.py
-------------
Custom embedding wrapper for nomic-ai/nomic-embed-text-v1.5.

This model requires different text prefixes for indexing vs retrieval:
  - "search_document: ..."  when embedding chunks stored in the database
  - "search_query: ..."     when embedding the user's question at query time

Using the correct prefix significantly improves retrieval accuracy.
The model is downloaded automatically on first use (~540 MB).
"""

from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class NomicEmbeddings(Embeddings):
    """
    Wraps nomic-ai/nomic-embed-text-v1.5 with the correct task prefixes.

    Usage:
        embeddings = NomicEmbeddings()
        # Embed chunks for storage:
        vectors = embeddings.embed_documents(["chunk 1", "chunk 2"])
        # Embed a query for retrieval:
        query_vec = embeddings.embed_query("How does zip() work?")
    """

    def __init__(self) -> None:
        self._model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = ["search_document: " + t for t in texts]
        return self._model.encode(prefixed, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(
            "search_query: " + text, normalize_embeddings=True
        ).tolist()
