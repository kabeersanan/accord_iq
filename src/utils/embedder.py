# src/utils/embedder.py

from typing import List, Dict
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    """
    Handles text embedding creation using a pre-trained SentenceTransformer model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Load once when class is created — avoids reloading model for every call
        self.model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")

    def create_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generates embeddings for a list of chunks.

        Args:
            chunks: List of dicts with keys {"id", "text", "metadata"}

        Returns:
            Same list with an added key "embedding" (numpy array)
        """
        texts = [chunk["text"] for chunk in chunks]

        print(f"Encoding {len(texts)} chunks...")
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Attach embeddings back to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = vectors[i]

        print("Embeddings generated successfully!")
        return chunks

    def embed_query(self, query: str) -> np.ndarray: #embedding for users's query
        """
        Creates an embedding for a single query string.
        """
        vec = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec
