from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _project_outputs() -> Path:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


class BiotechContextMemory:
    """
    Historical context for similar drug-class / sponsor patterns.
    Prefers Chroma when installed; falls back to JSONL + keyword overlap.
    """

    def __init__(self, collection_name: str = "biotech_alpha_context") -> None:
        self._collection_name = collection_name
        self._chroma = None
        self._collection = None
        self._fallback_path = _project_outputs() / "memory_fallback.jsonl"
        try:
            import chromadb  # type: ignore

            client = chromadb.PersistentClient(path=str(_project_outputs() / "chroma_db"))
            self._collection = client.get_or_create_collection(name=collection_name)
            self._chroma = chromadb
        except Exception:
            self._collection = None

    def _embed_text(self, text: str) -> List[float]:
        """Deterministic pseudo-embedding for fallback / tests (64-dim)."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        out: List[float] = []
        for i in range(0, len(h), 2):
            out.append(int.from_bytes(h[i : i + 2], "big") / 65535.0)
        while len(out) < 64:
            out.append(out[len(out) % len(out)] * 0.99)
        return out[:64]

    def upsert_context(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        meta = dict(metadata or {})
        meta["doc_id"] = doc_id
        if self._collection is not None:
            self._collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[{k: str(v) for k, v in meta.items()}],
            )
            return
        row = {"id": doc_id, "text": text, "metadata": meta}
        mode = "a" if self._fallback_path.exists() else "w"
        with open(self._fallback_path, mode, encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def query_similar(self, query_text: str, k: int = 3) -> List[Dict[str, Any]]:
        if self._collection is not None:
            res = self._collection.query(query_texts=[query_text], n_results=k)
            out: List[Dict[str, Any]] = []
            ids = (res.get("ids") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            dists = (res.get("distances") or [[]])[0] if res.get("distances") else [0.0] * len(ids)
            for i, doc_id in enumerate(ids):
                out.append(
                    {
                        "id": doc_id,
                        "text": (docs[i] if i < len(docs) else "")[:500],
                        "score": float(dists[i]) if i < len(dists) else 0.0,
                    }
                )
            return out

        if not self._fallback_path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with open(self._fallback_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        q_tokens = set(query_text.lower().split())
        scored: List[tuple[float, Dict[str, Any]]] = []
        for r in rows:
            text = r.get("text", "")
            t_tokens = set(text.lower().split())
            overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
            scored.append((overlap, r))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, r in scored[:k]:
            out.append(
                {
                    "id": r.get("id", ""),
                    "text": str(r.get("text", ""))[:500],
                    "score": float(score),
                }
            )
        return out
