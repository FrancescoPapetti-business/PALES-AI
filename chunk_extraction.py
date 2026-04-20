import os
import re
import json
import uuid
import pickle
from pathlib import Path

# === CONFIG ===
PROJECT_ROOT = r"C:\Users\adminAI\Documents\palesAI\classyfarm-rag"

INDEX_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "vector_store",
    "faiss_index"
)

PKL_PATH = os.path.join(INDEX_DIR, "index.pkl")

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "datasets",
    "v1_20251216_170818"
)

CHUNKS_PATH = os.path.join(DATASET_DIR, "chunks.jsonl")
PKL_PATH = os.path.join(INDEX_DIR, "index.pkl")

# === UTILS ===
def _first_present(d: dict, keys):
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            return d[k]
    return None

def _to_int(x, default=None):
    try:
        if x is None or isinstance(x, bool):
            return default
        if isinstance(x, int):
            return x
        if isinstance(x, float):
            return int(x)
        s = str(x).strip()
        if not s:
            return default
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else default
    except Exception:
        return default

def _basename(p):
    try:
        return Path(str(p)).name
    except Exception:
        return None

def _ext(p):
    try:
        return Path(str(p)).suffix
    except Exception:
        return None

def _stable_uuid(chunk_key: str) -> str:
    # UUIDv5 deterministico: stabile nel tempo per lo stesso chunk_key
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_key))

def _read_existing_ids(jsonl_path: str):
    ids = set()
    if not os.path.exists(jsonl_path):
        return ids
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("chunk_id")
                if cid:
                    ids.add(str(cid))
            except Exception:
                pass
    return ids

# === 1) carica docstore da index.pkl ===
with open(PKL_PATH, "rb") as f:
    payload = pickle.load(f)

docstore = payload[0]
index_to_docstore_id = payload[1]
docs_dict = getattr(docstore, "_dict", None)
if docs_dict is None:
    raise RuntimeError("docstore._dict non trovato: formato index.pkl inatteso.")

# === 2) leggi chunk_id già presenti nel chunks.jsonl full ===
existing_ids = _read_existing_ids(CHUNKS_PATH)

# === 3) costruisci nuovi record nello schema FULL ===
new_records = []

for faiss_id in range(len(index_to_docstore_id)):
    ds_id = index_to_docstore_id[faiss_id]
    d = docs_dict.get(ds_id)
    if d is None:
        continue

    md = d.metadata or {}
    text_content = d.page_content or ""

    # parent_doc_id / sha256: nel tuo schema è fondamentale
    sha256 = _first_present(md, ["sha256", "parent_doc_id", "doc_sha256"])
    if sha256 is None:
        # se non c'è sha, non puoi creare parent_doc_id coerente: skip (meglio che sporcare)
        continue
    sha256 = str(sha256)

    filename = _first_present(md, ["filename", "file_name"])
    if filename is None:
        src = _first_present(md, ["rel_path", "source", "file_path", "path"])
        filename = _basename(src) if src else None
    filename = filename or "unknown.pdf"

    rel_path = _first_present(md, ["rel_path"])
    extension = _first_present(md, ["extension"]) or _ext(filename) or ".pdf"

    page_number = _first_present(md, ["page_number", "page", "page_index"])
    page_number = _to_int(page_number, default=None)

    chunk_index = _first_present(md, ["chunk_index", "chunk_idx"])
    chunk_index = _to_int(chunk_index, default=None)
    if chunk_index is None:
        # fallback deterministico: se manca chunk_index, usa faiss_id (non ideale ma stabile)
        chunk_index = faiss_id

    # canonical_concepts
    canonical_concepts = md.get("canonical_concepts")
    if not isinstance(canonical_concepts, list):
        canonical_concepts = []

    # normalized_text: se già presente lo mantieni, altrimenti fallback = text_content lower-ish
    normalized_text = _first_present(md, ["normalized_text"])
    if normalized_text is None:
        normalized_text = text_content.strip().lower()

    # semantic_role / explicit / validated_at / source_type / topic / keywords / summary
    source_type = _first_present(md, ["source_type"]) or "unknown"
    topic = _first_present(md, ["topic"])
    keywords = md.get("keywords")
    if not isinstance(keywords, list):
        keywords = []
    summary = _first_present(md, ["summary"])
    semantic_role = _first_present(md, ["semantic_role"])
    explicit = md.get("explicit")
    if not isinstance(explicit, bool):
        explicit = False
    validated_at = md.get("validated_at")
    if not isinstance(validated_at, bool):
        # se viene dall'incremental, di solito non passa dalla validation: metti False
        validated_at = False

    # chunk_id: usa quello già presente, altrimenti UUIDv5 stabile su chiave deterministica
    chunk_id = _first_present(md, ["chunk_id"])
    if chunk_id is None:
        key = f"{sha256}|{filename}|{page_number}|{chunk_index}|{hash(normalized_text)}"
        chunk_id = _stable_uuid(key)
    else:
        chunk_id = str(chunk_id)

    if chunk_id in existing_ids:
        continue

    record = {
        "chunk_id": chunk_id,
        "parent_doc_id": sha256,
        "text_content": text_content,
        "canonical_concepts": canonical_concepts,
        "metadata": {
            # Mantieni e completa metadati per allineare lo schema full
            "filename": filename,
            "sha256": sha256,
            "parent_doc_id": sha256,
            "source_type": source_type,
            "extension": extension,
            "rel_path": rel_path,
            "summary": summary,
            "keywords": keywords,
            "topic": topic,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "semantic_role": semantic_role,
            "explicit": explicit,
            "canonical_concepts": canonical_concepts,
            "normalized_text": normalized_text,
            "validated_at": validated_at,
        }
    }

    new_records.append(record)

# === 4) merge append SAFE: backup + tmp + replace ===
os.makedirs(DATASET_DIR, exist_ok=True)
backup_path = CHUNKS_PATH + ".bak"
tmp_path = CHUNKS_PATH + ".tmp"

# backup una sola volta
if os.path.exists(CHUNKS_PATH) and not os.path.exists(backup_path):
    with open(CHUNKS_PATH, "rb") as src, open(backup_path, "wb") as dst:
        dst.write(src.read())

# copia su tmp
if os.path.exists(CHUNKS_PATH):
    with open(CHUNKS_PATH, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(src.read())
else:
    open(tmp_path, "wb").close()

# append nuovi
with open(tmp_path, "a", encoding="utf-8") as f:
    for r in new_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

os.replace(tmp_path, CHUNKS_PATH)

print("Aggiunti:", len(new_records))
print("Chunks file:", CHUNKS_PATH)
print("Backup:", backup_path if os.path.exists(backup_path) else "(non creato)")
