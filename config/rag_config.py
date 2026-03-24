from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class RagSettings:
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_collection: str = "medcity_documents"
    chroma_persist_dir: str = "models/vectorstore/chroma"
    top_k: int = 5

    @classmethod
    def from_env(cls) -> "RagSettings":
        current_file = Path(__file__).resolve()
        repo_root = current_file.parents[1]

        load_dotenv(repo_root / ".env", override=False)
        load_dotenv(current_file.parent / ".env", override=False)

        return cls(
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip(),
            chroma_collection=os.getenv("RAG_CHROMA_COLLECTION", "medcity_documents").strip(),
            chroma_persist_dir=os.getenv("RAG_CHROMA_PERSIST_DIR", "models/vectorstore/chroma").strip(),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _abs_persist_dir(persist_dir: str) -> str:
    path = Path(persist_dir)
    if not path.is_absolute():
        path = _repo_root() / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_embedding_model(settings: RagSettings | None = None) -> SentenceTransformer:
    cfg = settings or RagSettings.from_env()
    return _cached_embedding_model(cfg.embedding_model)


def build_chroma_client(settings: RagSettings | None = None) -> chromadb.PersistentClient:
    cfg = settings or RagSettings.from_env()
    persist_dir = _abs_persist_dir(cfg.chroma_persist_dir)
    return _cached_chroma_client(persist_dir)


@lru_cache(maxsize=4)
def _cached_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def _cached_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=persist_dir)


def build_chroma_collection(settings: RagSettings | None = None) -> Any:
    cfg = settings or RagSettings.from_env()
    client = build_chroma_client(cfg)
    return client.get_or_create_collection(name=cfg.chroma_collection)


def upsert_rag_documents(
    documents: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    ids: list[str] | None = None,
    settings: RagSettings | None = None,
) -> int:
    if not documents:
        return 0

    cfg = settings or RagSettings.from_env()
    collection = build_chroma_collection(cfg)
    model = build_embedding_model(cfg)

    final_ids = ids or [str(uuid.uuid4()) for _ in documents]
    embeddings = model.encode(documents).tolist()

    payload: dict[str, Any] = {
        "ids": final_ids,
        "documents": documents,
        "embeddings": embeddings,
    }
    if metadatas is not None:
        payload["metadatas"] = metadatas

    collection.upsert(**payload)
    return len(documents)


def retrieve_rag_context(
    query_text: str,
    settings: RagSettings | None = None,
    n_results: int | None = None,
) -> list[dict[str, Any]]:
    cfg = settings or RagSettings.from_env()
    collection = build_chroma_collection(cfg)
    model = build_embedding_model(cfg)

    top_k = n_results or cfg.top_k
    embedding = model.encode([query_text]).tolist()[0]

    result = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    items: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        items.append(
            {
                "document": doc,
                "metadata": metas[idx] if idx < len(metas) else {},
                "distance": distances[idx] if idx < len(distances) else None,
            }
        )
    return items
