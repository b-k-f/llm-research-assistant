"""Embeddings + vector database (Chroma, stored on disk)."""
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, persist_dir: Path, embedding_model: str, collection: str = "papers"):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self.model = SentenceTransformer(embedding_model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        # upsert = re-running ingestion on the same paper does not create duplicates
        self.collection.upsert(
            ids=ids, documents=texts, embeddings=self.embed(texts), metadatas=metadatas
        )

    def search(self, query: str, k: int) -> list[dict]:
        if self.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=self.embed([query]), n_results=min(k, self.count())
        )
        return [
            {"text": doc, "metadata": meta, "score": 1 - dist}
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def count(self) -> int:
        return self.collection.count()
