import logging
import re
from typing import List, Dict, Set
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger("ensemble_retriever")

# Stop words italiane generiche da escludere dal matching
_STOP_WORDS: Set[str] = {
    "cosa", "qual", "quale", "quali", "come", "dove", "quando",
    "perche", "sono", "essere", "avere", "fare", "dire", "vedere",
    "questo", "questa", "questi", "queste", "anche", "solo", "molto",
    "deve", "devo", "posso", "vuole", "vuoi", "faccio", "clicco",
    "non", "per", "che", "una", "uno", "degli", "delle", "nella",
    "dello", "della", "negli", "nelle", "dagli", "dalle", "sugli",
    "definizione", "chiamare", "significa", "intende", "vuole", "dire",
}

# Token non informativi nei filename da ignorare
_FILENAME_NOISE: Set[str] = {
    "pdf", "docx", "xlsx", "rev", "def", "min", "man", "classyfarm",
    "2019", "2020", "2021", "2022", "2023", "2024", "2025",
    "01", "02", "03", "vers", "ver", "final", "draft",
}


def preprocess_for_bm25(text: str) -> str:
    """
    Preprocessing per BM25:
    - Lowercase
    - Rimuovi TUTTI gli apostrofi/virgolette
    - Rimuovi punteggiatura
    - Normalizza spazi
    """
    if not text:
        return ""

    text = text.lower()

    # Rimuovi apostrofi e virgolette (tutti i tipi)
    apostrophes = "''`ʼʻ'\"„""«»"
    for char in apostrophes:
        text = text.replace(char, '')

    # Rimuovi punteggiatura
    punctuation = '.,;:!?()[]{}—–-_'
    for char in punctuation:
        text = text.replace(char, ' ')

    # Normalizza spazi
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _extract_query_keywords(query: str) -> Set[str]:
    """
    Estrae le keyword significative dalla query,
    rimuovendo stop words e token troppo corti.
    Usata da tutti i boost — calcolata UNA sola volta per query.
    """
    query_clean = preprocess_for_bm25(query)
    return {
        w for w in query_clean.split()
        if len(w) > 3 and w not in _STOP_WORDS
    }


class EnsembleRetriever(BaseRetriever, BaseModel):
    """
    Ensemble Retriever compatibile con LangChain 0.2.x
    Combina più retriever (FAISS, BM25, ecc.) con ranking pesato.

    BOOST attivi:
    - BOOST #1: Pattern definizionale nel contenuto (+80% / +40%)
    - BOOST #2: Chunk corti ad alta densità keyword (+30%)
    - BOOST #3: Filename relevance — automatico, senza hardcoding (+35% / +20%)

    HARD FILTERS:
    - Scarta sanctions_table se non richieste
    - Scarta checklist se query non è procedurale
    - Penalità -50% su chunk lunghi senza keyword match
    """

    retrievers: List[BaseRetriever] = Field(...)
    weights: List[float] = Field(...)
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"

    # ==============================
    # CORE METHOD (OBBLIGATORIO)
    # ==============================
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        if len(self.retrievers) != len(self.weights):
            logger.error("Mismatch between retrievers and weights count.")
            return []

        # ── Pre-calcola le keyword UNA SOLA VOLTA per tutti i boost ──
        query_keywords: Set[str] = _extract_query_keywords(query)
        query_lower = query.lower()
        is_definitional = (
            "definizione" in query_lower
            or "cos'è" in query_lower
            or "cosa è" in query_lower
            or "cosa sono" in query_lower
            or "cosa si intende" in query_lower
            or "per cosa sta" in query_lower
        )

        # ── 1. Fetch Results con Query Preprocessing ──
        results_by_retriever = []

        for i, retriever in enumerate(self.retrievers):
            try:
                from langchain_community.retrievers.bm25 import BM25Retriever

                if isinstance(retriever, BM25Retriever):
                    processed_query = preprocess_for_bm25(query)
                    logger.info(f"🔤 BM25 query preprocessed: '{query}' → '{processed_query}'")
                    query_to_use = processed_query
                else:
                    query_to_use = query

                if hasattr(retriever, "invoke"):
                    docs = retriever.invoke(query_to_use)
                elif hasattr(retriever, "get_relevant_documents"):
                    docs = retriever.get_relevant_documents(query_to_use)
                else:
                    docs = []

                results_by_retriever.append(docs)
                logger.info(f"   Retriever {i} ({type(retriever).__name__}): {len(docs)} docs")

            except Exception as e:
                logger.error(f"Retriever {i} failed: {e}")
                results_by_retriever.append([])

        # ── 2. Score con Boost ──
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        def _hash_doc(d: Document) -> str:
            return str(hash(d.page_content + str(d.metadata.get("chunk_id", ""))))

        for r_idx, docs in enumerate(results_by_retriever):
            weight = self.weights[r_idx]
            for rank, doc in enumerate(docs):
                did = _hash_doc(doc)
                if did not in doc_map:
                    doc_map[did] = doc

                # Score base decadente per rank
                rank_score = max(0.0, 1.0 - (rank * 0.05))

                content = doc.page_content
                content_lower = content.lower()

                # ── BOOST #1: Pattern definizionale nel contenuto ──
                if is_definitional and query_keywords:
                    for keyword in query_keywords:
                        # Pattern 1: "Keyword: ..." (definizione esplicita)
                        pattern_def = rf'\b{re.escape(keyword)}\w*\s*(dell\w+)?\s*:'
                        if re.search(pattern_def, content_lower):
                            rank_score *= 1.5
                            logger.info(f"   🎯 BOOST#1 definitional pattern ×1.8: '{keyword}:' in chunk")
                            break

                        # Pattern 2: Keyword all'inizio del chunk
                        if content_lower.strip().startswith(keyword):
                            rank_score *= 1.2
                            logger.info(f"   🎯 BOOST#1 definition start ×1.4: chunk starts with '{keyword}'")
                            break

                # ── BOOST #2: Chunk corti ad alta densità keyword ──
                if len(content) < 500 and query_keywords:
                    content_words = set(preprocess_for_bm25(content).split())
                    matches = len(query_keywords & content_words)
                    if matches >= 2:
                        density = matches / len(query_keywords)
                        if density > 0.5:
                            rank_score *= 1.3
                            logger.info(f"   🎯 BOOST#2 short+dense ×1.3: {matches}/{len(query_keywords)} keywords")

                # ── BOOST #3: Filename relevance (automatico) ──
                filename = doc.metadata.get("filename", "")
                if filename and query_keywords:
                    filename_tokens = set(preprocess_for_bm25(filename).split())
                    filename_tokens -= _FILENAME_NOISE
                    filename_hits = len(query_keywords & filename_tokens)

                    if filename_hits >= 2:
                        rank_score *= 1.45
                        logger.info(f"   🎯 BOOST#3 filename ×1.35: {filename_hits} hits in '{filename}'")
                    elif filename_hits == 1:
                        rank_score *= 1.30
                        logger.info(f"   🎯 BOOST#3 filename ×1.20: 1 hit in '{filename}'")

                current_score = doc_scores.get(did, 0.0)
                doc_scores[did] = current_score + (rank_score * weight)

        # ── 3. Filtering & Sorting ──
        ranked_results = []

        for did, score in doc_scores.items():
            doc = doc_map[did]
            meta = doc.metadata or {}
            content = doc.page_content

            # HARD FILTER A: tabelle sanzioni
            if meta.get("chunk_category") == "sanctions_table":
                if "sanzion" not in query_lower:
                    continue

            # HARD FILTER B: checklist
            if meta.get("chunk_category") == "checklist":
                if "controll" not in query_lower and "check" not in query_lower:
                    continue

            # PENALITÀ C: chunk lunghi senza keyword match
            if len(content) > 2000:
                q_terms = {t for t in query_lower.split() if len(t) > 3}
                if q_terms and not any(t in content_lower for t in q_terms):
                    score *= 0.5

            ranked_results.append((doc, score))

        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # ── 4. Logging Debug ──
        logger.info(f"🧩 Ensemble Query: '{query}'")
        logger.info(f"   Keywords estratte: {query_keywords}")
        logger.info(f"   Total unique docs after merge: {len(ranked_results)}")
        for i, (d, s) in enumerate(ranked_results[:3]):
            src = d.metadata.get("filename", "unknown")
            logger.info(f"   [{i+1}] Score: {s:.3f} | {src} | {d.page_content[:50]}...")

        return [d for d, s in ranked_results[:self.top_k]]

    # ==============================
    # COMPAT LAYERS
    # ==============================
    def retrieve(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)