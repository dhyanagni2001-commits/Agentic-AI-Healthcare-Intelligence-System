"""Sentence-transformers embedding wrapper — singleton model load.

Model: all-MiniLM-L6-v2 (384-dim). Lazy-loaded so importing this module
doesn't pay the model-load cost until embeddings are actually needed
(both the FastAPI process and any standalone ingestion script import the
same module without double-loading the model).
"""
from __future__ import annotations
import logging
from typing import List
import numpy as np

# torch must be imported before faiss anywhere in the process — on this
# platform (macOS + faiss-cpu + torch's bundled libomp) importing faiss
# first causes a libomp symbol conflict that segfaults at interpreter exit.
# Importing torch here, eagerly, guarantees the safe order as long as this
# module is imported before any `import faiss` (vector_store.py does this).
import torch  # noqa: F401

log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info(f"Loading embedding model '{MODEL_NAME}'…")
        _model = SentenceTransformer(MODEL_NAME)
        log.info("Embedding model ready")
    return _model


def encode(texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Encode a batch of texts to float32 vectors, L2-normalized by default
    so downstream inner-product search behaves as cosine similarity."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")
    model = _get_model()
    vecs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=normalize, show_progress_bar=False,
    )
    return vecs.astype("float32")


def encode_one(text: str, normalize: bool = True) -> np.ndarray:
    return encode([text], normalize=normalize)[0]
