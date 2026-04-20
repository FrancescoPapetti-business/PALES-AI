#!/usr/bin/env python3
"""
retrieval_eval_openai.py

Valuta il retrieval su un FAISS locale usando OpenAIEmbeddings come provider di embedding.

Requisiti:
 - avere OPENAI_API_KEY impostata nell'ambiente
 - langchain_openai e langchain_community installati

Nota:
 Assicurati che il modello di embedding usato qui (default: text-embedding-3-small) sia lo stesso usato per costruire l'indice FAISS,
 altrimenti l'embed-dimension potrebbe non corrispondere e il caricamento o le query potrebbero fallire.

Uso:
  $env:OPENAI_API_KEY="sk-..."     # PowerShell
  python validation/retrieval_eval_openai.py --gold validation/gold_dataset.jsonl --faiss-path data/vector_store/faiss_index --ks 1 3 5 --out results/retrieval_report.jsonl
  set OPENAI_API_KEY=sk-...        # CMD
  python validation/retrieval_eval_openai.py --gold validation/gold_dataset.jsonl --faiss-path data/vector_store/faiss_index --ks 1 3 5 --out results/retrieval_report.jsonl
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional
import logging
from tqdm import tqdm
import sys
import os
from dotenv import load_dotenv
load_dotenv()  # This loads your .env at project root, making your key available.

# Logging semplice
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrieval_eval_openai")

# Robust import of Embeddings
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    try:
        from langchain_openai.embeddings import OpenAIEmbeddings
    except Exception as e:
        OpenAIEmbeddings = None
        logger.error("Cannot import OpenAIEmbeddings! Install 'langchain-openai'. Error: %s", e)
        sys.exit(1)

# Import FAISS vectorstore
try:
    from langchain_community.vectorstores import FAISS
except Exception as e:
    FAISS = None
    logger.error("Impossibile importare langchain_community.vectorstores.FAISS: %s", e)

def load_gold(gold_path: Path) -> List[Dict[str, Any]]:
    data = []
    if not gold_path.exists():
        logger.error(f"File gold dataset mancante: {gold_path}")
        sys.exit(8)
    with gold_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def doc_to_id(doc: Any) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    filename = meta.get("filename") or meta.get("source") or meta.get("rel_path") or "UNKNOWN"
    page = meta.get("page_number") or meta.get("page") or meta.get("page_idx") or None
    if page is None:
        return f"{filename}"
    try:
        p = int(page)
        return f"{filename}::p{p}"
    except Exception:
        return f"{filename}::p{page}"

def first_relevant_rank(retrieved_ids: List[str], relevant_ids: Iterable[str], max_k: int) -> Optional[int]:
    rel_set = set(relevant_ids or [])
    for i, rid in enumerate(retrieved_ids[:max_k], start=1):
        if rid in rel_set:
            return i
        fname = rid.split("::")[0]
        if fname in rel_set:
            return i
    return None

def normalize_relevants(case: Dict[str, Any]) -> List[str]:
    relevants = case.get("relevants_doc_ids") or case.get("expected_supporting_docs") or []
    normalized = []
    if relevants and isinstance(relevants[0], dict):
        for d in relevants:
            fn = d.get("filename") or d.get("rel_path") or ""
            pg = d.get("page") or d.get("page_number") or d.get("page_idx")
            if pg:
                normalized.append(f"{fn}::p{pg}")
            else:
                normalized.append(fn)
    else:
        normalized = list(relevants)
    return normalized

def evaluate_queries(gold: List[Dict[str, Any]], retriever, ks: List[int], max_k: int) -> Dict[str, Any]:
    metrics = {k: {"recall_count": 0, "mrr_sum": 0.0} for k in ks}
    coverage_count = 0
    total = len(gold)
    per_query = []

    for case in tqdm(gold, desc="Evaluating"):
        qid = case.get("id")
        question = case.get("question") or ""
        role = case.get("role")
        retrieval_query = question if not role else f"{question} [role: {role}]"

        try:
            if hasattr(retriever, "get_relevant_documents"):
                docs = retriever.get_relevant_documents(retrieval_query)
            elif hasattr(retriever, "retrieve"):
                docs = retriever.retrieve(retrieval_query)
            elif hasattr(retriever, "search"):
                docs = retriever.search(retrieval_query)
            else:
                raise RuntimeError("Retriever non compatibile: manca metodo di invocazione.")
        except Exception as e:
            logger.error("Errore retrieval per %s: %s", qid, e)
            docs = []

        retrieved_ids = [doc_to_id(d) for d in docs]
        normalized_relevants = normalize_relevants(case)

        rank_first = first_relevant_rank(retrieved_ids, normalized_relevants, max_k)
        if rank_first is not None:
            coverage_count += 1

        for k in ks:
            rank_k = first_relevant_rank(retrieved_ids, normalized_relevants, k)
            if rank_k is not None:
                metrics[k]["recall_count"] += 1
                metrics[k]["mrr_sum"] += 1.0 / rank_k

        per_query.append({
            "id": qid,
            "question": question,
            f"retrieved_top{max_k}": retrieved_ids[:max_k],
            "relevants": normalized_relevants,
            "first_relevant_rank": rank_first
        })

    aggregated = {}
    for k in ks:
        aggregated[f"Recall@{k}"] = round(metrics[k]["recall_count"] / total if total else 0.0, 4)
        aggregated[f"MRR@{k}"] = round(metrics[k]["mrr_sum"] / total if total else 0.0, 4)
    aggregated[f"coverage@{max_k}"] = round(coverage_count / total if total else 0.0, 4)
    aggregated["total_queries"] = total
    return {"aggregated": aggregated, "per_query": per_query}

def main():
    p = argparse.ArgumentParser(description="Retrieval evaluation (OpenAIEmbeddings-only)")
    p.add_argument("--gold", type=Path, required=True, help="Path to gold JSONL")
    p.add_argument("--faiss-path", type=Path, required=True, help="Path to FAISS index directory")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 10], help="K values for Recall@K / MRR@K")
    p.add_argument("--embed-model", type=str, default="text-embedding-3-small", help="OpenAI embedding model name to use (must match the one used to build index)")
    p.add_argument("--out", type=Path, default=None, help="Output JSONL per-query report path")
    p.add_argument("--max-k", type=int, default=10, help="Max K to consider when computing coverage")
    args = p.parse_args()

    # Pre-check: imports and env
    if FAISS is None:
        logger.error("FAISS non disponibile. Installa langchain_community e dipendenze relative.")
        sys.exit(2)
    if OpenAIEmbeddings is None:
        logger.error("OpenAIEmbeddings non disponibile. Installa langchain_openai.")
        sys.exit(3)
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY non trovata nell'ambiente. Impostala prima di eseguire.")
        sys.exit(4)

    gold = load_gold(args.gold)
    if not gold:
        logger.error("Gold dataset vuoto o file non valido.")
        sys.exit(5)

    ks = sorted(set(args.ks))
    max_k = max(args.max_k, max(ks))

    try:
        emb = OpenAIEmbeddings(model=args.embed_model)
        logger.info("OpenAIEmbeddings inizializzato con modello: %s", args.embed_model)
    except Exception as e:
        logger.exception("Errore inizializzazione OpenAIEmbeddings: %s", e)
        sys.exit(6)

    try:
        vector_store = FAISS.load_local(str(args.faiss_path), emb)
        retriever = vector_store.as_retriever(search_kwargs={"k": max_k})
        logger.info("FAISS index caricato correttamente da: %s", args.faiss_path)
    except Exception as e:
        logger.exception("Errore caricamento FAISS: %s", e)
        sys.exit(7)

    results = evaluate_queries(gold, retriever, ks, max_k)

    print("\n=== Retrieval Evaluation Aggregated Metrics ===")
    for k, v in results["aggregated"].items():
        print(f"{k}: {v}")

    if args.out:
        outp = args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fh:
            for r in results["per_query"]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info("Per-query report salvato in: %s", outp)

if __name__ == "__main__":
    main()