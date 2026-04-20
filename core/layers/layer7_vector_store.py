"""
layer7_vector_store.py
Carica i Golden Datasets (JSONL) e crea l'indice vettoriale FAISS + BM25 (Hybrid).
Rispetta i chunk semantici creati nel Layer 4.

BUG-06 MVP:
- retrieval 2-stadi (Recall largo -> Rerank deterministico)
- K dinamico per intent procedurale
- boost chunk procedurali + penalty chunk lunghi descrittivi
"""

import sys
import os
import json
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.bm25 import BM25Retriever


from core.layers.layer6_dataset_manager import get_dataset_fingerprint

# ===========================================================
# ROOT + ENV (robusto: risale finché non trova ".env")
# ===========================================================

current = Path(__file__).resolve()
for parent in current.parents:
    if (parent / ".env").exists():
        BASE_DIR = parent
        break
else:
    raise RuntimeError("❌ Impossibile trovare il file .env nella struttura delle cartelle.")

ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

print(f"[DEBUG L7] Layer loaded .env from: {ENV_PATH}")


# ===========================================================
# PATHS
# ===========================================================

DATASETS_DIR = BASE_DIR / "data" / "datasets"
INDEX_DIR = BASE_DIR / "data" / "vector_store"
FAISS_DIR = INDEX_DIR / "faiss_index"
BM25_PATH = INDEX_DIR / "bm25_retriever.pkl"  # Legacy pickle path
BM25_JSON_PATH = INDEX_DIR / "bm25_documents.json"  # New safe format

# Configurazione Embedding
EMBEDDING_MODEL = "text-embedding-3-small"


# ===========================================================
# BUG-06 CONFIG (ranking deterministico)
# ===========================================================

PROCEDURAL_ROLES = {
    "mandatory_rule",
    "access_rule",
    "service_mapping",
    "contact_rule",
}

PROCEDURAL_INTENT_KEYWORDS = [
    # supporto/contatto
    "assistenza", "supporto", "contattare", "contatto", "email", "telefono", "helpdesk",
    "orari", "orario", "ticket",
    # accesso/permessi/procedura
    "accesso", "accedere", "richiesta", "inviare", "invio", "procedura", "procedura guidata",
    "abilitazione", "permesso", "ruolo", "valutatore", "utente", "credenziali",
    # regole/obblighi/limiti
    "obbligatorio", "deve", "non deve", "vietato", "limite", "scadenza", "entro", "termine",
]

# moltiplicatori (semplici, spiegabili, tuning facile)
BOOST_PROCEDURAL = 1.30
PENALTY_LONG_DESCRIPTIVE = 0.90
LONG_WORD_THRESHOLD = 700

# retrieval settings
# MODIFICATO: k aumentato da 5 a 8 per migliorare Context Recall
# Data modifica: 2026-02-05
DEFAULT_TOP_K = 8
PROCEDURAL_TOP_K = 12

# stage1: quanti candidati tiro su prima del rerank
# (più alto di top_k per fare competizione "vera" nel rerank)
STAGE1_CANDIDATES = 40


# ===========================================================
# HELPERS
# ===========================================================

def sanitize_metadata(md: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rende i metadata sicuri per persistenza/serializzazione.
    - tipi primitivi: ok
    - list: convertite in stringa
    - dict: json.dumps
    - altri tipi: str()
    """
    safe_md: Dict[str, Any] = {}
    for k, v in (md or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            safe_md[k] = v
        elif isinstance(v, list):
            safe_md[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
            safe_md[k] = json.dumps(v, ensure_ascii=False)
        else:
            safe_md[k] = str(v)
    return safe_md


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def word_count(text: str) -> int:
    t = normalize_text(text)
    if not t:
        return 0
    return len(t.split())


# ===========================================================
# CANONICAL TEXT BUILDER — Bug-02
# ===========================================================

def build_index_text(text: str, metadata: Dict[str, Any]) -> str:
    """
    Costruisce il testo da indicizzare:
    [CANONICAL_CONCEPT] [CANONICAL_CONCEPT] testo originale

    Usa SOLO informazioni già prodotte dal Layer 4.
    """
    if not text:
        return ""

    canonical = metadata.get("canonical_concepts")

    if isinstance(canonical, str):
        canonical = [c.strip() for c in canonical.split(",") if c.strip()]

    if not canonical:
        return text

    prefix = " ".join(f"[{c}]" for c in canonical)
    return f"{prefix} {text}".strip()


# ===========================================================
# INTENT (procedural vs non)
# ===========================================================

def infer_query_intent(query: str) -> str:
    q = normalize_text(query)
    if not q:
        return "generic"

    hits = 0
    for kw in PROCEDURAL_INTENT_KEYWORDS:
        if kw in q:
            hits += 1

    # soglia minima: basta poco per spostare K solo su procedural
    if hits >= 1:
        return "procedural"

    return "generic"


def resolve_top_k(query: str, default_k: int = DEFAULT_TOP_K, procedural_k: int = PROCEDURAL_TOP_K) -> int:
    intent = infer_query_intent(query)
    return procedural_k if intent == "procedural" else default_k


# ===========================================================
# LOAD DATASET
# ===========================================================

def load_latest_dataset() -> List[Document]:
    """
    Trova l'ultima versione del dataset (JSONL) e la converte in Document LangChain.
    NON rifà lo splitting: usa i chunk semantici del Layer 4.
    Atteso: data/datasets/vX/chunks.jsonl
    """
    if not DATASETS_DIR.exists():
        print(f"⚠ Directory datasets non trovata: {DATASETS_DIR}")
        return []

    versions = [d for d in DATASETS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not versions:
        print("⚠ Nessun dataset versionato trovato.")
        return []

    latest_version = sorted(versions, key=lambda x: x.name, reverse=True)[0]
    jsonl_path = latest_version / "chunks.jsonl"

    if not jsonl_path.exists():
        print(f"⚠ File chunks.jsonl non trovato in: {jsonl_path}")
        return []

    print(f"📂 Caricamento dataset versione: {latest_version.name}")

    documents: List[Document] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)

                raw_md = {
                    "chunk_id": data.get("chunk_id"),
                    "parent_doc_id": data.get("parent_doc_id"),
                    **(data.get("metadata") or {}),
                }

                text_content = data.get("text_content", "") or ""
                safe_md = sanitize_metadata(raw_md)
                
                # FILTERING POLICY (Layer 7)
                # 1. Scarta chunk troppo lunghi descrittivi (rumore per retrieval)
                if len(text_content) > 3000:
                    # print(f"   [L7 Filter] Skip chunk troppo lungo ({len(text_content)} chars)")
                    continue
                
                # 2. Scarta tabelle sanzioni/checklist se marcate da Layer 5
                cat = safe_md.get("chunk_category")
                if cat in ["sanctions_table", "checklist"]:
                    # print(f"   [L7 Filter] Skip category: {cat}")
                    continue

                # BUG-06: persistiamo info utile al ranking
                safe_md["word_count"] = word_count(text_content)

                index_text = build_index_text(text_content, safe_md)
                doc = Document(
                    page_content=index_text,
                    metadata=safe_md,
                )
                documents.append(doc)

    except Exception as e:
        print(f"❌ Errore lettura JSONL: {e}")
        return []

    print(f"   → Caricati {len(documents)} documenti (chunk atomici).")
    return documents


# ===========================================================
# CREATE INDICES
# ===========================================================

def create_vector_store(docs: List[Document]) -> None:
    """Crea e salva l'indice FAISS su disco."""
    if not docs:
        print("⚠ Nessun documento da indicizzare.")
        return

    print(f"🧠 Calcolo Embeddings ({EMBEDDING_MODEL})...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = FAISS.from_documents(docs, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(FAISS_DIR))

    print(f"✅ Indice FAISS salvato in: {FAISS_DIR}")


def create_bm25_index(docs: List[Document]) -> None:
    """
    Crea un indice BM25 (Sparse) per ricerca keywords.
    Salva i documenti come JSON (sicuro, portabile).
    Il BM25Retriever viene ricostruito al caricamento.
    """
    if not docs:
        print("⚠ Nessun documento per BM25.")
        return

    print("🔍 Creazione Indice BM25 (Keyword Search)...")

    # Serializza documenti come JSON (sicuro, no pickle)
    docs_data = {
        "format_version": "4.0",
        "dataset_fingerprint": get_dataset_fingerprint(),
        "created_at": datetime.now().isoformat(),
        "default_k": 15,
        "documents": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata or {}
            }
            for doc in docs
        ]
    }

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    with open(BM25_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, ensure_ascii=False)

    print(f"✅ Indice BM25 salvato in: {BM25_JSON_PATH}")
    print(f"   • Documenti: {len(docs)}")
    print(f"   • Formato: JSON v4.0 (sicuro)")


# ===========================================================
# LOAD INDICES (runtime)
# ===========================================================

def load_faiss_vector_store() -> FAISS:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if not FAISS_DIR.exists():
        raise RuntimeError(f"❌ FAISS index non trovato: {FAISS_DIR}. Esegui full_indexing_pipeline()")
    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_bm25_index() -> Optional[Any]:
    """
    Carica BM25 retriever.
    Priorita': JSON v4.0 (sicuro) > pickle legacy (fallback).
    """
    # ========================================
    # PRIORITA' 1: Formato JSON v4.0 (sicuro)
    # ========================================
    if BM25_JSON_PATH.exists():
        try:
            with open(BM25_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            doc_list = data.get("documents", [])
            if not doc_list:
                print("❌ BM25 JSON vuoto")
                return None

            docs = [
                Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                for d in doc_list
            ]

            retriever = BM25Retriever.from_documents(docs)
            retriever.k = data.get("default_k", 15)

            print(f"✅ BM25 caricato (JSON v{data.get('format_version', '4.0')})")
            print(f"   • Documenti: {len(docs)}")
            print(f"   • Default k: {retriever.k}")

            return retriever

        except Exception as e:
            print(f"❌ Errore caricamento BM25 JSON: {e}")
            import traceback
            traceback.print_exc()

    # ========================================
    # PRIORITA' 2: Pickle legacy (fallback + migrazione)
    # ========================================
    if BM25_PATH.exists():
        import pickle
        print("⚠️ BM25 pickle legacy rilevato - migrazione a JSON...")

        try:
            with open(BM25_PATH, 'rb') as f:
                data = pickle.load(f)

            # Formato v3.0 (retriever completo)
            if isinstance(data, dict) and "retriever" in data:
                retriever = data["retriever"]
                if hasattr(retriever, 'docs'):
                    # Migra a JSON
                    create_bm25_index(retriever.docs)
                    print("✅ Migrazione pickle -> JSON completata")
                return retriever

            # Formato v2.0 (texts + metadatas)
            elif isinstance(data, dict) and "texts" in data:
                texts = data.get("texts", [])
                metadatas = data.get("metadatas", [])
                docs = [
                    Document(page_content=text, metadata=meta)
                    for text, meta in zip(texts, metadatas)
                ]
                retriever = BM25Retriever.from_documents(docs)
                retriever.k = 15
                # Migra a JSON
                create_bm25_index(docs)
                print(f"✅ BM25 ricostruito da legacy v2.0 ({len(docs)} docs) e migrato a JSON")
                return retriever

            else:
                print(f"❌ BM25 pickle formato sconosciuto: {type(data)}")
                return None

        except Exception as e:
            print(f"❌ Errore caricamento BM25 pickle: {e}")
            import traceback
            traceback.print_exc()
            return None

    print(f"⚠️ BM25 index non trovato in: {BM25_JSON_PATH}")
    return None


# ===========================================================
# BUG-06: HYBRID RETRIEVAL (2-stadi)
# ===========================================================

def _parse_semantic_role(md: Dict[str, Any]) -> str:
    # atteso: md["semantic_role"] come stringa; se manca -> "unknown"
    role = (md or {}).get("semantic_role")
    return str(role).strip() if role else "unknown"


def _parse_word_count(md: Dict[str, Any]) -> int:
    try:
        return int((md or {}).get("word_count") or 0)
    except Exception:
        return 0


def _parse_canonical_concepts(md: Dict[str, Any]) -> List[str]:
    cc = (md or {}).get("canonical_concepts")
    if not cc:
        return []
    if isinstance(cc, str):
        # può essere "a, b, c" o json stringato
        # primo tentativo: split su virgola
        items = [x.strip() for x in cc.split(",") if x.strip()]
        return items
    if isinstance(cc, list):
        return [str(x).strip() for x in cc if str(x).strip()]
    return []


def _concept_match_boost(query: str, md: Dict[str, Any]) -> float:
    """
    Boost leggero se query contiene canonical concept.
    (Non sostituisce i tag nel testo: è un "tie-breaker" a favore del segnale.)
    """
    q = normalize_text(query)
    if not q:
        return 1.0
    concepts = _parse_canonical_concepts(md)
    if not concepts:
        return 1.0

    hits = 0
    for c in concepts:
        c_norm = normalize_text(c)
        if c_norm and c_norm in q:
            hits += 1

    if hits >= 2:
        return 1.10
    if hits == 1:
        return 1.05
    return 1.0


def rerank_score(
    similarity_score: float,
    doc: Document,
    query: str,
) -> float:
    """
    similarity_score: più alto = più rilevante (vedi conversione sotto)
    Applica boost/penalty deterministici per BUG-06.
    """
    md = doc.metadata or {}
    role = _parse_semantic_role(md)
    wc = _parse_word_count(md)

    score = similarity_score

    # (6.2) Boost chunk procedurali "giusti"
    if role in PROCEDURAL_ROLES:
        score *= BOOST_PROCEDURAL

    # tie-break: canonical concepts match
    score *= _concept_match_boost(query, md)

    # (6.3) Penalizza chunk lunghi e descrittivi
    if wc > LONG_WORD_THRESHOLD and role == "descriptive":
        score *= PENALTY_LONG_DESCRIPTIVE

    return score


def retrieve_hybrid(
    query: str,
    top_k: Optional[int] = None,
    stage1_candidates: int = STAGE1_CANDIDATES,
) -> List[Document]:
    """
    Retrieval a due stadi:
      Stadio 1: recall largo (FAISS + BM25)
      Stadio 2: rerank deterministico (semantic_role, word_count, canonical_concepts)
    """
    if top_k is None:
        top_k = resolve_top_k(query)

    vector_store = load_faiss_vector_store()
    bm25 = load_bm25_index() # Ora ritorna Optional
    
    if not bm25:
        print("⚠️ BM25 non disponibile, fallback su solo FAISS.")
        # Fallback logica solo FAISS... (omesso per brevità, il focus è sul fix)

    # -------------------------
    # STADIO 1 — Recall largo
    # -------------------------

    # FAISS: usa similarity_search_with_score (in FAISS il valore è spesso distanza; convertiamo in "più alto=meglio")
    faiss_hits: List[Tuple[Document, float]] = []
    try:
        raw = vector_store.similarity_search_with_score(query, k=stage1_candidates)
        # tipicamente score = distanza, più basso è meglio -> convertiamo a similarity "alta = meglio"
        for doc, dist in raw:
            sim = 1.0 / (1.0 + float(dist))  # stabile, monotona
            faiss_hits.append((doc, sim))
    except Exception as e:
        print(f"⚠ FAISS retrieval failed: {e}")

    # BM25: non dà score, quindi assegniamo un base-score decrescente per posizione
    bm25_docs = []
    if bm25:
        try:
            bm25.k = stage1_candidates
            bm25_docs = bm25.get_relevant_documents(query)
        except Exception as e:
            print(f"⚠ BM25 retrieval failed: {e}")

    bm25_hits: List[Tuple[Document, float]] = []
    for rank, d in enumerate(bm25_docs):
        # rank 0 più alto, poi decresce
        bm25_hits.append((d, 0.60 * (1.0 / (1.0 + rank))))

    # merge candidati per chunk_id (se presente), altrimenti per hash del contenuto
    merged: Dict[str, Tuple[Document, float]] = {}

    def _key(doc: Document) -> str:
        md = doc.metadata or {}
        cid = md.get("chunk_id")
        if cid:
            return f"chunk:{cid}"
        return f"txt:{hash(doc.page_content)}"

    for doc, s in faiss_hits + bm25_hits:
        kkey = _key(doc)
        if kkey not in merged:
            merged[kkey] = (doc, s)
        else:
            # somma morbida: se un doc arriva forte da entrambi, sale
            prev_doc, prev_s = merged[kkey]
            merged[kkey] = (prev_doc, prev_s + s)

    candidates = list(merged.values())

    # -------------------------
    # STADIO 2 — Rerank
    # -------------------------
    reranked = sorted(
        candidates,
        key=lambda pair: rerank_score(pair[1], pair[0], query),
        reverse=True
    )

    return [d for d, _ in reranked[:top_k]]


# ===========================================================
# PIPELINE
# ===========================================================

def full_indexing_pipeline():
    """Orchestrator del Layer 7."""
    print("🚀 Avvio Pipeline Indicizzazione (Layer 7)...")

    docs = load_latest_dataset()
    if not docs:
        print("❌ Indicizzazione interrotta: nessun dato.")
        return

    create_vector_store(docs)
    create_bm25_index(docs)

    print("🏁 Pipeline Indicizzazione completata.")


def incremental_indexing_update(new_chunks: List[Dict[str, Any]], index_path: Optional[str] = None) -> str:
    """
    Aggiornamento incrementale dell'indice FAISS:
    aggiunge nuovi chunk senza ricostruire tutto da zero.

    new_chunks: lista di dict con:
      - "text_content": str
      - "metadata": dict
    """
    if index_path is None:
        index_path = str(FAISS_DIR)

    index_path = Path(index_path)

    print(f"🧠 Aggiornamento incrementale indice FAISS → {index_path}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    try:
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise RuntimeError(f"❌ Impossibile caricare indice FAISS: {e}")

    new_docs: List[Document] = []
    for c in new_chunks:
        text = (c.get("text_content") or "").strip()
        if not text:
            continue
        md = sanitize_metadata(c.get("metadata") or {})

        # BUG-06: coerenza con full indexing
        md["word_count"] = word_count(text)
        index_text = build_index_text(text, md)

        new_docs.append(Document(page_content=index_text, metadata=md))

    if not new_docs:
        print("⚠ Nessun chunk valido da aggiungere.")
        return str(index_path)

    vectorstore.add_documents(new_docs)
    vectorstore.save_local(str(index_path))

    print("✅ Indicizzazione incrementale completata.")
    return str(index_path)

def get_indexed_sha256() -> list[str]:
    """
    Ritorna l'elenco degli sha256 dei documenti già indicizzati.
    Legge i metadata direttamente dall'indice FAISS salvato su disco.
    """
    try:
        vectorstore = load_faiss_vector_store()

        if not hasattr(vectorstore, "docstore"):
            return []

        sha_set = set()
        for doc in vectorstore.docstore._dict.values():
            meta = getattr(doc, "metadata", {}) or {}
            sha = meta.get("sha256")
            if sha:
                sha_set.add(sha)

        return list(sha_set)

    except Exception as e:
        print(f"⚠️ Errore nel recupero sha256 indicizzati: {e}")
        return []


# ===========================================================
# MAIN (debug rapido)
# ===========================================================

if __name__ == "__main__":
    # 1) indicizza
    full_indexing_pipeline()

    # 2) smoke test retrieval (solo se vuoi verificarlo subito)
    try:
        q = "in che orari posso contattare l’assistenza?"
        docs = retrieve_hybrid(q)
        print("\n=== SMOKE TEST ===")
        print("Query:", q)
        for i, d in enumerate(docs, 1):
            role = (d.metadata or {}).get("semantic_role")
            wc = (d.metadata or {}).get("word_count")
            src = (d.metadata or {}).get("filename") or (d.metadata or {}).get("source") or "?"
            print(f"{i}. role={role} wc={wc} src={src}")
            print(d.page_content[:200], "...\n")
    except Exception as e:
        print(f"⚠ Smoke test retrieval skipped/failed: {e}")
