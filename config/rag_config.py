from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RagSettings:
    embedding_model: str = "all-MiniLM-L6-v2"
    pinecone_api_key: str = ""
    pinecone_index: str = "medcity"
    top_k: int = 5

    @classmethod
    def from_env(cls) -> "RagSettings":
        current_file = Path(__file__).resolve()
        repo_root = current_file.parents[1]

        load_dotenv(repo_root / ".env", override=False)
        load_dotenv(current_file.parent / ".env", override=False)

        return cls(
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip(),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
            pinecone_index=os.getenv("PINECONE_INDEX", "medcity").strip(),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
        )


def build_embedding_model(settings: RagSettings | None = None) -> SentenceTransformer:
    cfg = settings or RagSettings.from_env()
    return _cached_embedding_model(cfg.embedding_model)


@lru_cache(maxsize=4)
def _cached_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _build_pinecone_index(settings: RagSettings | None = None) -> Any:
    cfg = settings or RagSettings.from_env()
    if not cfg.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY no configurada en .env")
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    return pc.Index(cfg.pinecone_index)


def upsert_rag_documents(
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    ids: list[str] | None = None,
    settings: RagSettings | None = None,
) -> int:
    if not documents:
        return 0

    cfg = settings or RagSettings.from_env()
    index = _build_pinecone_index(cfg)
    model = build_embedding_model(cfg)

    final_ids = ids or [str(uuid.uuid4()) for _ in documents]
    embeddings = model.encode(documents).tolist()

    # Pinecone upsert en lotes de 100
    vectors = []
    for i, (vid, emb, doc) in enumerate(zip(final_ids, embeddings, documents)):
        meta = dict(metadatas[i]) if metadatas and i < len(metadatas) else {}
        # Pinecone metadata: guardar texto del documento (max 40KB por vector)
        meta["_document"] = doc[:39000]
        # Asegurar que todos los valores de metadata sean tipos soportados
        clean_meta = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
        vectors.append({"id": vid, "values": emb, "metadata": clean_meta})

    batch_size = 100
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch)

    return len(documents)


def retrieve_rag_context(
    query_text: str,
    settings: RagSettings | None = None,
    n_results: int | None = None,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = settings or RagSettings.from_env()
    index = _build_pinecone_index(cfg)
    model = build_embedding_model(cfg)

    top_k = n_results or cfg.top_k
    embedding = model.encode([query_text]).tolist()[0]

    query_kwargs: dict[str, Any] = {
        "vector": embedding,
        "top_k": top_k,
        "include_metadata": True,
    }

    # Convertir filtro ChromaDB-style a Pinecone filter
    if where:
        pc_filter: dict[str, Any] = {}
        for key, val in where.items():
            if isinstance(val, dict):
                # Ya es formato operador {"$eq": ...}
                pc_filter[key] = val
            else:
                pc_filter[key] = {"$eq": val}
        query_kwargs["filter"] = pc_filter

    result = index.query(**query_kwargs)

    items: list[dict[str, Any]] = []
    for match in result.get("matches", []):
        meta = dict(match.get("metadata", {}))
        doc_text = meta.pop("_document", "")
        score = match.get("score", 0)
        # Pinecone cosine: score 1.0 = idéntico, convertir a distancia
        distance = 1.0 - score
        items.append(
            {
                "document": doc_text,
                "metadata": meta,
                "distance": distance,
            }
        )
    return items
