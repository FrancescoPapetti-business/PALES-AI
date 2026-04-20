import os
import json
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"
BM25_OUT = VECTOR_DIR / "bm25_retriever.pkl"

# usa latest se esiste, altrimenti prendi l'ultima cartella v*
latest = DATASETS_DIR / "latest"
if latest.exists():
    chunks_path = latest / "chunks.jsonl"
else:
    versions = sorted([p for p in DATASETS_DIR.glob("v*") if p.is_dir()])
    if not versions:
        raise RuntimeError(f"Nessun dataset trovato in {DATASETS_DIR}")
    chunks_path = versions[-1] / "chunks.jsonl"

if not chunks_path.exists():
    raise FileNotFoundError(f"chunks.jsonl non trovato: {chunks_path}")

print(f"[INFO] Carico chunks da: {chunks_path}")

# BM25Retriever langchain_community
from langchain_community.retrievers.bm25 import BM25Retriever

texts = []
metas = []

with chunks_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # il tuo schema L6 di solito ha "chunk_text" o "text"
        txt = obj.get("text_content") or obj.get("chunk_text") or obj.get("text") or obj.get("content")
        txt = (
            obj.get("text_content")
            or obj.get("chunk_text")
            or obj.get("text")
            or obj.get("content")
            or (obj.get("metadata", {}) or {}).get("normalized_text")
        )
        if not txt:
            continue
        texts.append(txt)
        metas.append(obj.get("metadata", {}))

if not texts:
    raise RuntimeError("Nessun testo caricato dai chunk (campi chunk_text/text/content vuoti).")

print(f"[INFO] #chunks caricati: {len(texts)}")

retriever = BM25Retriever.from_texts(texts=texts, metadatas=metas)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

with BM25_OUT.open("wb") as f:
    pickle.dump(retriever, f)

print(f"[OK] Salvato BM25 in: {BM25_OUT}")