"""
Motor de busca vetorial: armazena embeddings no ChromaDB e recupera chunks relevantes.
Usa o modelo all-MiniLM-L6-v2 via ChromaDB (funciona em PT por ser multilíngue).
"""

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from typing import Any
from src.extractor import Chunk


COLLECTION_NAME = "ajudante_docs"


class VectorStore:
    def __init__(self, persist_dir: str = "data/chroma"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ef = DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        existing_ids = set(self._collection.get(ids=[c.chunk_id for c in chunks])["ids"])
        new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]
        if not new_chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in new_chunks],
            documents=[c.text for c in new_chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in new_chunks],
        )

    def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        results = self._collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "score": round(1 - dist, 4),  # cosine similarity
            })
        return hits

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
