"""FAISS vector store: flat inner-product index + an id->metadata sidecar.

FAISS only stores vectors; it knows nothing about hospitals, doc types, or
text. This wraps an IndexFlatIP (inner product on L2-normalized vectors =
cosine similarity) with a parallel metadata list, and persists both to
disk so the server doesn't recompute ~250K embeddings on every restart.

Index choice: IndexFlatIP, not IVF. At this corpus size (low hundreds of
thousands of vectors), flat search is single-digit milliseconds even on a
free-tier EC2 instance — IVF's sub-linear search only pays off above
roughly 1M+ vectors, and it adds a training step and an nlist
hyperparameter that would only hurt recall here for no latency benefit.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.ingestion.documents import SearchableDocument
from backend.services.embedding_service import EMBEDDING_DIM, encode

log = logging.getLogger(__name__)

DEFAULT_DIR = Path(__file__).parent.parent.parent / "data" / "vector_index"


class FaissVectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self._index = None
        self._docs: List[SearchableDocument] = []

    @property
    def total(self) -> int:
        return len(self._docs)

    def build(self, docs: List[SearchableDocument], batch_size: int = 256) -> None:
        import faiss
        self._docs = list(docs)
        self._index = faiss.IndexFlatIP(self.dim)
        texts = [d.text for d in self._docs]
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vecs = encode(batch, normalize=True)
            if vecs.shape[0]:
                self._index.add(vecs)
        log.info(f"Built FAISS index: {self._index.ntotal} vectors, dim={self.dim}")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[SearchableDocument, float]]:
        if self._index is None or self._index.ntotal == 0:
            return []
        qvec = encode([query], normalize=True)
        scores, idxs = self._index.search(qvec, min(top_k, self._index.ntotal))
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._docs):
                continue
            out.append((self._docs[idx], float(score)))
        return out

    def save(self, directory: Path = DEFAULT_DIR) -> None:
        import faiss
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        meta = [d.to_dict() for d in self._docs]
        (directory / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False))
        log.info(f"Saved vector store to {directory} ({len(self._docs)} docs)")

    def load(self, directory: Path = DEFAULT_DIR) -> bool:
        import faiss
        index_path = directory / "index.faiss"
        meta_path = directory / "metadata.json"
        if not index_path.exists() or not meta_path.exists():
            return False
        self._index = faiss.read_index(str(index_path))
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        self._docs = [SearchableDocument(**d) for d in raw]
        log.info(f"Loaded vector store from {directory} ({len(self._docs)} docs)")
        return True
