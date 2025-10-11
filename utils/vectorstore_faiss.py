# utils/vectorstore_faiss.py
import os
import json
import numpy as np

class FaissVectorStore:
    def __init__(self, dim=384, metric="cosine"):
        """
        metric: 'cosine' or 'l2'
        """
        import faiss
        self.dim = dim
        self.metric = metric.lower()
        self._normalize = self.metric == "cosine"
        self.index = faiss.IndexFlatIP(dim) if self._normalize else faiss.IndexFlatL2(dim)
        self.id_map = {}
        self._next_id = 0
        print(f"FAISS index initialized with dimension {dim}")

    def _maybe_normalize(self, arr: np.ndarray):
        """Normalize vectors if cosine metric."""
        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr.astype("float32")

    def add_embeddings(self, chunks):
        """Add chunk embeddings to the FAISS index."""
        if not chunks:
            return
        vecs = []
        for c in chunks:
            emb = np.array(c["embedding"], dtype="float32")
            if emb.shape[0] != self.dim:
                raise ValueError(f"Embedding dim mismatch: {emb.shape[0]} != {self.dim}")
            vecs.append(emb)
            self.id_map[str(self._next_id)] = {
                "id": c["id"],
                "text": c["text"],
                "metadata": c.get("metadata", {})
            }
            self._next_id += 1

        mat = np.vstack(vecs)
        mat = self._maybe_normalize(mat)
        self.index.add(mat)
        print(f"Added {len(vecs)} vectors. Total now: {self.index.ntotal}")

    def search(self, query_vec, top_k=5):
        """Return the top_k most similar chunks for a query vector."""
        if isinstance(query_vec, list):
            query_vec = np.array(query_vec, dtype="float32")
        q = query_vec.reshape(1, -1)
        q = self._maybe_normalize(q)

        D, I = self.index.search(q, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            doc = self.id_map.get(str(idx))
            if not doc:
                continue
            score = float(dist) if self._normalize else float(1.0 / (1.0 + dist))
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": score
            })
        return results
