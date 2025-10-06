# src/utils/vectorstore_faiss.py

import faiss
import numpy as np
from typing import List, Dict

class FaissVectorStore:
    """
    Handles storing and retrieving text embeddings using FAISS.
    """

    def __init__(self, dim: int):
        """
        Initialize FAISS index.

        Args:
            dim (int): Dimension of embeddings 
        """
        # Using an IndexFlatIP means inner product = cosine similarity (since vectors are normalized)
        self.index = faiss.IndexFlatIP(dim)
        self.id_map = {}  # maps FAISS index -> metadata
        print(f"FAISS index initialized with dimension {dim}")

    def add_embeddings(self, chunks: List[Dict]):
        """
        Add embeddings and metadata to FAISS index.

        Args:
            chunks: list of dicts with keys: id, embedding, metadata
        """
        vectors = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")

        start_idx = self.index.ntotal
        self.index.add(vectors)

        for i, chunk in enumerate(chunks):
            self.id_map[start_idx + i] = {
                "id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            }

        print(f"Added {len(chunks)} vectors. Total now: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict]:
        """
        Search for the most similar vectors.

        Args:
            query_vector: numpy array of query embedding
            top_k: number of closest results to return

        Returns:
            List of metadata dicts for top matches.
        """
        query_vector = np.expand_dims(query_vector.astype("float32"), axis=0)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            item = self.id_map[idx].copy()
            item["score"] = float(score)
            results.append(item)
        return results
