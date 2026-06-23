from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Tuple

# Global cache for embedding model
_embedding_model = None


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Get cached embedding model (loaded only once).
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        Cached SentenceTransformer model instance
        
    Raises:
        RuntimeError: If model cannot be loaded
    """
    global _embedding_model
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {str(e)}")
    return _embedding_model


def encode_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    vectors = model.encode(texts)
    arr = np.array(vectors, dtype="float32")
    return arr


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatL2:
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def build_index_from_documents(model: SentenceTransformer, docs: List[dict]) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    """Build FAISS index from documents using provided model."""
    texts = [d["text"] for d in docs]
    vectors = encode_texts(model, texts)
    index = build_faiss_index(vectors)
    return index, vectors


def search_index(index: faiss.IndexFlatL2, model: SentenceTransformer, query: str, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    qv = model.encode(query)
    qarr = np.array([qv], dtype="float32")
    distances, indices = index.search(qarr, k)
    return distances, indices


def retrieve_context_with_pages(indices: np.ndarray, docs: List[dict]) -> str:
    """Given FAISS indices result and original docs, return concatenated context with page markers."""
    context = ""
    for idx in indices[0]:
        page = docs[int(idx)]["page"]
        context += f"\n[PAGE {page}]\n"
        context += docs[int(idx)]["text"]
        context += "\n\n"
    return context
