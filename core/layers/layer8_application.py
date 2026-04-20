"""
Layer 8 – Application Logic (RAG Engine)
------------------------------------------
Orchestrator stateless che gestisce:
1. Safety (Input/Output)
2. Retrieval Ibrido (Vector + Keyword)
3. Costruzione Prompt (con Memory e System Prompt esterno)
4. Generazione LLM
"""

import sys
import os
import time
import hashlib
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from cachetools import TTLCache

# Setup Root e Env
# Trova la ROOT del progetto salendo finché non trova ".env"
current = Path(__file__).resolve()

for parent in current.parents:
    if (parent / ".env").exists():
        BASE_DIR = parent
        break
else:
    raise RuntimeError("❌ Impossibile trovare il file .env nella struttura delle cartelle.")

ENV_PATH = BASE_DIR / ".env"
print(f"🔍 ENV caricato da: {ENV_PATH}")  # Debug utilissimo
load_dotenv(ENV_PATH)

# Aggiungi la root al PYTHONPATH
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Recupero ibrido
from core.layers.layer6_dataset_manager import get_dataset_fingerprint
from core.layers.layer7_vector_store import load_bm25_index # Import corretto
from core.custom.ensemble_retriever import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever

# Core
from langchain_core.documents import Document

# Moduli Custom
from core.prompts.system_prompts import get_system_prompt
from core.layers.layer4_semantic_rewriting import normalize_concepts, inject_canonical_tags
from core.layers.layer4_semantic_rewriting import normalize_concepts, inject_canonical_tags, expand_acronym_query
from core.layers.layer_safety_moderation import SafetyHandler, is_out_of_domain
from utils.logging_config import get_logger, get_kpi_logger
from core.layers.layer7_vector_store import load_bm25_index, resolve_top_k

# --- CONFIGURAZIONE ---
logger = get_logger("layer8_engine")
kpi_logger = get_kpi_logger()

INDEX_DIR = BASE_DIR / "data" / "vector_store"
FAISS_PATH = INDEX_DIR / "faiss_index"
BM25_PATH = INDEX_DIR / "bm25_retriever.pkl"

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
EMBED_MODEL = "text-embedding-3-small"

MAX_CTX_CHARS = int(os.getenv("MAX_CTX_CHARS", "6000"))
DEFAULT_TOP_K = 8

# Cache TTL per risposte (max 200 entry, scade dopo 1 ora)
_response_cache = TTLCache(maxsize=200, ttl=3600)

# ==========================
#   CROSS-DOCUMENT EVIDENCE
# ==========================
import re
from collections import defaultdict
from utils.temporal_parser import (
    extract_temporal_entities, 
    validate_temporal_preservation,
    enrich_context_with_temporal_note
)
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"\b(\+?\d[\d\s.-]{6,}\d)\b")
TIME_RE = re.compile(r"\b(\d+)\s*(ore?|giorni?|settimane?|mesi?|gg\.?|h)\b", re.IGNORECASE)

def extract_atoms(text: str) -> dict:
    return {
        "emails": set(EMAIL_RE.findall(text or "")),
        "phones": set(PHONE_RE.findall(text or "")),
        "times": set(TIME_RE.findall(text or "")),
    }

def is_procedural_question(question: str) -> bool:
    triggers = [
        "come", "procedura", "può", "deve", "è possibile",
        "invio", "richiesta", "registrazione", "accesso"
    ]
    q = question.lower()
    return any(t in q for t in triggers)

def infer_intent(question: str) -> str:
    """
    Classifica l'intent della domanda per routing ottimizzato.
    
    PRIORITÀ:
    1. Procedural (come faccio, come posso, scaricare, modificare...)
    2. Acronym definition (BDN, OdC, SQNBA...)
    3. Definitional (cos'è, cosa significa, cosa mi dice, cosa vedo, a cosa serve..)
    4. Generic (fallback)
    """
    import re  # Import locale per sicurezza
    
    q_lower = question.lower()
    
    # ========================================
    # PRIORITÀ 1: PROCEDURAL
    # ========================================
    # Frasi procedurali complete
    procedural_phrases = [
        "come faccio", "come posso", "come si", "come fare",
        "come devo", "come richiedo", "come seleziono",
        "come accedo", "come mi registro", "come registro",
        "come inserisco", "come compilo", "come invio",
        "dove trovo", "dove posso", "dove devo",
        "chi puo'", "chi può", "chi deve",
        "cosa devo fare", "cosa serve per",
        "in caso di",
    ]
    
    if any(phrase in q_lower for phrase in procedural_phrases):
        return "procedural"
    
    # Verbi d'azione (solo se c'è "come" nella frase)
    if "come" in q_lower:
        procedural_verbs = [
            "scaricare", "modificare", "compilare", "associare",
            "inserire", "creare", "eliminare", "cancellare",
            "richiedere", "selezionare", "continuare", "finire",
            "caricare", "aggiornare", "visualizzare", "vedere",
            "accedere", "registrare", "registrarmi", "abilitare",
            "delegare", "inviare", "compilare", "ottenere",
        ]
        
        if any(verb in q_lower for verb in procedural_verbs):
            return "procedural"
    
    # ========================================
    # 2. ACRONYM DEFINITION
    # ========================================
    # Pattern per rilevare acronimi (supporta maiuscole miste come "OdC")
    acronym_patterns = [
        (r'\bper\s+cosa\s+sta\s+([A-Z][a-zA-Z]{1,5})\b', re.IGNORECASE),
        (r'\bcosa\s+significa\s+([A-Z][a-zA-Z]{1,5})\b', re.IGNORECASE),
        (r'\bcosa\s+si\s+intende\s+per\s+([A-Z][a-zA-Z]{1,5})\b', re.IGNORECASE),
        (r'\bcosa\s+vuol\s+dire\s+([A-Z][a-zA-Z]{1,5})\b', re.IGNORECASE),
    ]

    for pattern, flags in acronym_patterns:
        match = re.search(pattern, question, flags)
        if match and match.groups():
            acronym = match.group(1)
            
            # Verifica che sia un acronimo:
            # - Lunghezza 2-6 caratteri
            # - Almeno 2 maiuscole (es. "OdC", "BDN", "SQNBA")
            # - Oppure tutto maiuscolo
            upper_count = sum(1 for c in acronym if c.isupper())
            
            if len(acronym) >= 2 and (acronym.isupper() or upper_count >= 2):
                return "acronym_definition"
    
    # ========================================
    # PRIORITÀ 3: DEFINITIONAL
    # ========================================
    definitional_keywords = [
        "cos'è", "cosa è", "che cos'è",
        "definizione", "significato",
        "cosa sono", "cosa vogliono dire",
        "qual è", "quali sono", "che cosa",
        # Forme descrittive
        "cosa mi dice", "cosa mi mostra", "cosa vedo",
        "cosa indica", "cosa mostra", "cosa rappresenta",
        "cosa contiene", "cosa include", "cosa riporta",
        "a cosa serve", "a cosa corrisponde",
        "come si interpreta", "come si legge",
    ]
    
    if any(keyword in q_lower for keyword in definitional_keywords):
        return "definitional"
    
    # ========================================
    # FALLBACK: GENERIC
    # ========================================
    return "generic"

def collect_evidence(docs: List[Document]) -> dict:
    """
    Raggruppa i Document per (canonical_concept, procedure_scope)
    """
    assert all(hasattr(d, "metadata") for d in docs), \
    "collect_evidence expects Document list"

    evidence = defaultdict(list)

    for d in docs:
        meta = d.metadata or {}
        concepts = meta.get("canonical_concepts") or []

        # NORMALIZZAZIONE FORZATA
        if isinstance(concepts, str):
            # gestisce "A, B, C"
            concepts = [c.strip() for c in concepts.split(",") if c.strip()]

        if not isinstance(concepts, list):
            concepts = []


        scope = meta.get("procedure_scope")

        for c in concepts:
            key = (c, scope)
            evidence[key].append(d)

    return evidence

def is_sufficient(evidence: dict) -> dict:
    for docs in evidence.values():
        assert all(hasattr(d, "page_content") for d in docs), \
    "is_sufficient expects Document list"

    winners = {}

    for key, docs in evidence.items():
        if len(docs) < 2:
            continue

        atoms = {"emails": set(), "phones": set(), "times": set()}
        for d in docs:
            a = extract_atoms(d.page_content)
            for k in atoms:
                atoms[k] |= a[k]

        # conflitto se stesso campo ha più valori
        conflict = any(len(v) > 1 for v in atoms.values())
        if not conflict:
            winners[key] = docs

    return winners

def synthesize_answer_from_evidence(groups: dict) -> tuple[str, list]:
    """
    Sintesi deterministica:
    - Estrae SOLO righe con atomi (email, telefono, orari)
    - Usa SOLO Document validi
    - Nessuna inferenza LLM
    """

    answer_parts: list[str] = []
    used_docs: list = []

    for _, docs in groups.items():
        if not isinstance(docs, list):
            continue

        seen_lines = set()

        for d in docs:
            if not hasattr(d, "page_content"):
                continue

            for line in d.page_content.split("\n"):
                l = line.strip()
                if not l or l in seen_lines:
                    continue

                atoms = extract_atoms(l)
                if any(atoms.values()):
                    answer_parts.append(l)
                    seen_lines.add(l)

            if d not in used_docs:
                used_docs.append(d)

    # Log diagnostico se vuoto
    if not answer_parts:
        logger.warning("⚠️ Atom Extraction Failed: Nessun atomo (email/tel/orari) trovato nei documenti candidati.")
        for d in used_docs:
             logger.debug(f"   - Doc analizzato: {d.metadata.get('filename')} (Category: {d.metadata.get('chunk_category')})")


    return "\n".join(answer_parts), used_docs

def synthesize_procedure_from_docs(docs: list[Document], question: str) -> str:
    """
    Sintesi procedurale guidata:
    - NON risponde solo sì/no
    - Ricostruisce i passaggi usando i documenti
    - Linguaggio prudente e spiegativo
    """

    context = "\n\n".join(d.page_content for d in docs if hasattr(d, "page_content"))

    prompt = f"""
Sei PALES-AI, assistente della piattaforma ClassyFarm. Rispondi SOLO con i documenti forniti.

DOMANDA: {question}

DOCUMENTAZIONE:
{context}

ISTRUZIONI PER LA RISPOSTA:
1. Spiega la procedura PASSO PER PASSO (1. 2. 3.)
2. Per ogni passaggio specifica ESATTAMENTE:
   - Dove andare (menu, sezione, pagina, pulsante)
   - Cosa fare (cliccare, compilare, selezionare, allegare)
   - Cosa serve (SPID, CIE, codice fiscale, documenti, moduli)
3. Se la procedura richiede un modulo, indica QUALE modulo e DOVE scaricarlo
4. Se la procedura richiede invio email, specifica A CHI e COSA allegare
5. Se serve un prerequisito (es. registrazione a BDN, SPID), menzionalo all'inizio
6. Se il documento indica ruoli diversi (operatore, veterinario, delegato), specifica per chi vale la procedura
7. NON inventare passaggi non presenti nei documenti
8. Se la procedura non e' completa nei documenti, dillo chiaramente

RISPOSTA:
"""

    # Usa lo stesso LLM del motore
    return prompt


def is_directly_answerable(question: str, docs: List[Document]) -> List[Document]:
    
        q = question.lower()
        candidates = []

        for d in docs:
            text = d.page_content.lower()
            atoms = extract_atoms(text)

        # Orari
            if "orari" in q or "quando" in q:
                if atoms["times"]:
                    candidates.append(d)

        # Contatti / assistenza
            if "assistenza" in q or "contattare" in q:
                if atoms["phones"] or atoms["emails"]:
                    candidates.append(d)

        # Accesso / visibilità
            if "visualizza" in q or "allevamenti" in q:
                if "visualizza" in text or "esclusivamente" in text:
                    candidates.append(d)

        return candidates

def is_context_usable(docs: List[Document], min_chars: int = 150) -> bool:
    """
    Verifica deterministica: se esiste almeno un documento con contenuto sufficiente,
    il contesto è considerato utilizzabile e il chatbot DEVE rispondere.
    """
    if not docs:
        return False
    return any(len(d.page_content.strip()) >= min_chars for d in docs)

def should_exclude_doc(meta: dict, text: str, intent: str) -> bool:
    """
    Filtro intelligente: non scarta FAQ/Checklist se l'intent è procedurale.
    """
    doc_type = meta.get("doc_type")
    
    # Sanzioni sempre escluse (rumore)
    if doc_type == "sanzioni":
        return True
        
    # Checklist/FAQ spesso contengono le risposte operative (Q7, Q12)
    if doc_type == "checklist":
        if intent in ["procedural", "generic", "definitional"]:
            return False
            
    # Tabelle: escludi solo se molto rumorose
    if doc_type == "table":
        if text.count("|") > 10 or text.count(";") > 10:
            return True
            
    return False

def is_refusal(text: str) -> bool:
    """Rileva frasi di rifiuto standard."""
    t = text.lower()
    triggers = [
        "non emerge", "non è specificato", "non risulta", 
        "non ho trovato informazioni", "mi dispiace", 
        "dalla documentazione", "non viene menzionato"
    ]
    # Check semplice
    if any(tr in t for tr in triggers):
        return True
    return False

# --- TEMPLATE DI ASSEMBLAGGIO ---
# Questo template unisce i pezzi, ma il testo "core" viene da get_system_prompt()
FULL_PROMPT_TEMPLATE = """
{system_prompt_text}

=== CONTESTO DOCUMENTALE ===
I seguenti documenti sono stati recuperati dalla knowledge base di ClassyFarm.
Usali come UNICA fonte per la tua risposta.

{context_str}

=== LINK E RISORSE UTILI ===
{links_str}

=== STORICO CONVERSAZIONE ===
{chat_history_str}

=== DOMANDA DELL'UTENTE ===
{question}

=== COME RISPONDERE ===
1. Rispondi basandoti ESCLUSIVAMENTE sui documenti sopra
2. Se i documenti contengono la risposta (anche parziale), RISPONDI con quello che hai
3. Se un documento sembra incompleto, dillo: "In base alla documentazione disponibile..."
4. Se i documenti NON contengono la risposta, dillo chiaramente e suggerisci di contattare l'assistenza ClassyFarm (800 08 22 80 o helpdesk@classyfarm.it)
5. Per procedure: usa passaggi numerati, specifica menu/sezioni/pulsanti dove cliccare
6. Se ci sono link utili nei documenti, menzionali nella risposta
7. Sii conciso ma completo — non omettere dettagli operativi importanti
"""

class RAGEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGEngine, cls).__new__(cls)
            try:
                cls._instance._initialize()
            except Exception as e:
                cls._instance = None
                raise e
        return cls._instance

    def _initialize(self):
        logger.info("🚀 Init RAG Engine Layer 8...")
        
        self.safety = SafetyHandler()
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1, request_timeout=30)
        self.embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
        
        # ========================================
        # CARICA FAISS
        # ========================================
        faiss_path = BASE_DIR / "data" / "vector_store" / "faiss_index"
        
        self.faiss_store = FAISS.load_local(
            str(faiss_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        faiss_retriever = self.faiss_store.as_retriever(
            search_kwargs={"k": DEFAULT_TOP_K}
        )
        
        # ========================================
        # CARICA BM25 (CON FIX)
        # ========================================
        try:
            # Import funzione di caricamento dal Layer 7
            from core.layers.layer7_vector_store import load_bm25_index
            
            bm25_retriever = load_bm25_index()
            
            if bm25_retriever is not None:
                # Verifica che sia un oggetto valido
                if hasattr(bm25_retriever, 'k'):
                    bm25_retriever.k = DEFAULT_TOP_K
                    
                    # Crea ensemble con pesi
                    ensemble_weights = [0.5, 0.5]  # 50% FAISS, 50% BM25
                    
                    self.retriever = EnsembleRetriever(
                        retrievers=[faiss_retriever, bm25_retriever],
                        weights=ensemble_weights,
                        top_k=DEFAULT_TOP_K
                    )
                    
                    self.bm25_active = True
                    logger.info("✅ BM25 Retriever caricato e attivo.")
                
                else:
                    logger.error("❌ BM25 retriever non valido - usa solo FAISS")
                    self.retriever = faiss_retriever
                    self.bm25_active = False
            
            else:
                logger.warning("⚠️ BM25 non disponibile - uso solo FAISS")
                self.retriever = faiss_retriever
                self.bm25_active = False
        
        except Exception as e:
            logger.error(f"❌ Errore caricamento BM25: {e}")
            import traceback
            traceback.print_exc()
            self.retriever = faiss_retriever
            self.bm25_active = False
    
        logger.info(f"✅ RAG Engine inizializzato (BM25: {self.bm25_active})")

def format_sources_from_docs(docs: list) -> list[dict]:
    """
    Accetta SOLO LangChain Document.
    Qualsiasi altro tipo viene ignorato.
    """
    out = []
    seen = set()

    if not isinstance(docs, list):
        return out

    for doc in docs:
        if not hasattr(doc, "metadata"):
            continue

        meta = doc.metadata or {}
        if not isinstance(meta, dict):
            continue

        filename = meta.get("filename")
        page = meta.get("page_number")

        if not filename:
            continue

        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "filename": filename,
            "page_number": page,
            "rel_path": meta.get("rel_path"),
            "sha256": meta.get("sha256"),
            "source_url": meta.get("source_url", ""),
        })

    return out

def format_links_for_prompt(docs: List[Any]) -> str:
    all_links = {}

    for doc in docs:
        links = (doc.metadata or {}).get("extracted_links", [])

        # se qualcuno ha salvato una stringa singola invece di lista
        if isinstance(links, str):
            links = [links]

        for link in links:
            # supporta sia dict che stringa
            if isinstance(link, str):
                url = link.strip()
                description = url
                link_type = "url"
            elif isinstance(link, dict):
                url = (link.get("url") or "").strip()
                description = (link.get("description") or url).strip()
                link_type = (link.get("type") or "url").strip()
            else:
                continue

            if not url:
                continue

            if url not in all_links:
                if link_type == "email" or url.lower().startswith("mailto:"):
                    if not url.lower().startswith("mailto:"):
                        url = f"mailto:{url}"
                    markdown_link = f" - [Contatto Email: {url.replace('mailto:', '')}]({url})"
                else:
                    markdown_link = f" - [Link Esterno: {description}]({url})"
                all_links[url] = markdown_link

    if not all_links:
        return "Nessun link o risorsa esterna pertinente è stata recuperata."

    return "\n".join(all_links.values())

def format_chat_history(history: List[Dict[str, str]]) -> str:
    if not history: return "Nessuno."
    # Limita a ultime 5 coppie per evitare superamento context window
    recent = history[-5:]
    return "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent])

def format_docs_for_prompt(docs: list) -> str:
    """
    Converte i Document recuperati in un contesto testuale da passare al prompt.
    """
    assert all(hasattr(d, "page_content") for d in docs), \
        "format_docs_for_prompt expects LangChain Document objects"
    out = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        filename = meta.get("filename", "Sconosciuto")
        page = meta.get("page_number")

        header = f"[{i}] {filename}"
        if page:
            header += f" (pag. {page})"
        source_url = meta.get("source_url", "")
        if source_url:
            header += f" — Fonte: {source_url}"

        out.append(header)
        out.append(d.page_content.strip())
        out.append("")  # riga vuota

    return "\n".join(out).strip()

def _retrieve_docs(retriever, query: str, *, k: int | None = None) -> List[Document]:
    """
    Compat wrapper: prova in ordine invoke -> get_relevant_documents -> retrieve -> search.
    Logga il metodo usato e normalizza output a List[Document].
    """
    def _ensure_list(out):
        if out is None:
            return []
        if isinstance(out, list):
            return out
        return [out]

    # 1) invoke (LangChain 0.2.x)
    if hasattr(retriever, "invoke"):
        try:
            out = retriever.invoke(query)
            out = _ensure_list(out)
            logger.info(f"[RETRIEVE] method=invoke docs={len(out)} type={type(retriever)}")
            return out
        except Exception as e:
            logger.warning(f"[RETRIEVE] invoke() failed: {e}")

    # 2) legacy
    if hasattr(retriever, "get_relevant_documents"):
        try:
            out = retriever.get_relevant_documents(query)
            out = _ensure_list(out)
            logger.info(f"[RETRIEVE] method=get_relevant_documents docs={len(out)} type={type(retriever)}")
            return out
        except Exception as e:
            logger.warning(f"[RETRIEVE] get_relevant_documents() failed: {e}")

    # 3) custom
    if hasattr(retriever, "retrieve"):
        try:
            out = retriever.retrieve(query, k=k) if k is not None else retriever.retrieve(query)
            out = _ensure_list(out)
            logger.info(f"[RETRIEVE] method=retrieve docs={len(out)} type={type(retriever)}")
            return out
        except Exception as e:
            logger.warning(f"[RETRIEVE] retrieve() failed: {e}")

    # 4) vectorstore style
    if hasattr(retriever, "search"):
        try:
            out = retriever.search(query, k=k) if k is not None else retriever.search(query)
            out = _ensure_list(out)
            logger.info(f"[RETRIEVE] method=search docs={len(out)} type={type(retriever)}")
            return out
        except Exception as e:
            logger.warning(f"[RETRIEVE] search() failed: {e}")

    raise AttributeError(f"Retriever has no supported retrieval method. Type={type(retriever)}")


def process_request(question: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
    """
    Process user question with RAG pipeline.
    
    Pipeline:
    1. Safety check + normalization
    2. Intent classification + canonical concepts
    3. Retrieval (FAISS + BM25 ensemble)
    4. Multi-strategy answer generation
    5. Safety output validation
    """
    engine = RAGEngine()
    start_time = time.time()

    # ========================================
    # 0. CACHE LOOKUP
    # ========================================
    cache_key = hashlib.md5(question.encode()).hexdigest()
    cached = _response_cache.get(cache_key)
    if cached:
        logger.info(f"⚡ Cache hit per query (key={cache_key[:8]})")
        cached["latency"] = round(time.time() - start_time, 3)
        return cached

    # ========================================
    # 1. SAFETY CHECK & NORMALIZATION
    # ========================================
    safe_q, flags, err_msg = engine.safety.sanitize_input(question)
    
    if flags["prompt_injection"] or flags["toxic_content"]:
        logger.warning(f"🚨 Safety block: {err_msg}")
        return {
            "answer": "Richiesta bloccata per motivi di sicurezza.",
            "error": err_msg,
            "latency": 0,
            "sources": [],
            "retrieved_contexts": [],
            "safety_flags": flags
        }
    
    # Canonical concepts normalization
    norm_q = normalize_concepts(safe_q)
    canonical_q = norm_q.get("canonical_concepts", []) or []
    
    # ========================================
    # 2. INTENT CLASSIFICATION & QUERY EXPANSION
    # ========================================

    intent = infer_intent(safe_q)
    logger.info(f"🧭 Intent: {intent} | Canonical: {canonical_q}")
    # Dynamic top_k based on intent (uses layer7 resolve_top_k)
    FINAL_TOP_K = resolve_top_k(safe_q)
    DEFAULT_TOP_K_DYNAMIC = FINAL_TOP_K
    # ----------------------------------------
    # 2.1 CANONICAL TAG INJECTION (existing)
    # ----------------------------------------
    retrieval_q = inject_canonical_tags(safe_q, canonical_q)

    if intent == "acronym_definition":
        retrieval_q = expand_acronym_query(retrieval_q)
        logger.info(f"📝 Acronym query: {retrieval_q}")

    # ----------------------------------------
    # 2.2 ADVANCED QUERY EXPANSION ⭐ NEW
    # ----------------------------------------

    from core.layers.layer2_query_expansion import QueryExpander

    # Configuration
    EXPANSION_ENABLED = True  # Toggle query expansion
    MAX_EXPANDED_QUERIES = 5  # Max queries (including original)
    EXPANSION_STRATEGIES = ["synonyms", "intent"]  # Skip "llm" for speed

    if EXPANSION_ENABLED:
        logger.info("🔍 Query Expansion enabled")
        
        # Initialize expander (lazy singleton)
        if not hasattr(engine, '_query_expander'):
            engine._query_expander = QueryExpander(llm=engine.llm)
            logger.info("✅ QueryExpander initialized")
        
        # Expand query
        try:
            expansion_results = engine._query_expander.expand(
                query=retrieval_q,  # Use canonical-enriched query
                intent=intent,
                strategies=EXPANSION_STRATEGIES
            )
            
            expanded_queries = expansion_results["all"][:MAX_EXPANDED_QUERIES]
            
            logger.info(f"📊 Expanded: 1 → {len(expanded_queries)} queries")
            for i, eq in enumerate(expanded_queries, 1):
                preview = eq[:60] + "..." if len(eq) > 60 else eq
                logger.info(f"  {i}. {preview}")
            
            # Use expanded queries for retrieval
            retrieval_queries = expanded_queries
        
        except Exception as e:
            logger.error(f"❌ Query expansion error: {e} → using original")
            retrieval_queries = [retrieval_q]

    else:
        logger.info("⚠️ Query Expansion DISABLED")
        retrieval_queries = [retrieval_q]

    # ----------------------------------------
    # 2.3 DYNAMIC ENSEMBLE WEIGHTS TUNING
    # ----------------------------------------

    if engine.bm25_active and isinstance(engine.retriever, EnsembleRetriever):
        # Intent-based weight optimization
        weight_map = {
            "acronym_definition": [0.3, 0.7],  # Heavy BM25 (exact acronym match)
            "definitional": [0.6, 0.4],         # Moderate BM25 (keyword definitions)
            "procedural": [0.7, 0.3],           # Heavy FAISS (semantic steps)
            "generic": [0.6, 0.4]               # Balanced (default)
        }
        
        new_weights = weight_map.get(intent, [0.6, 0.4])  # Default to balanced
        engine.retriever.weights = new_weights
        
        logger.info(f"⚖️  Ensemble weights: FAISS={new_weights[0]:.1f} | BM25={new_weights[1]:.1f}")

    else:
        logger.info("📊 Single retriever mode (no ensemble)")

    # ========================================
    # 3. RETRIEVAL WITH RE-RANKING
    # ========================================

    from core.layers.layer7_reranker import CrossEncoderReranker, deduplicate_documents

    # Initialize reranker (lazy singleton)
    if not hasattr(engine, '_reranker'):
        logger.info("🔧 Initializing CrossEncoderReranker...")
        try:
            engine._reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cpu"
            )
            logger.info("✅ Reranker initialized")
        except Exception as e:
            logger.error(f"❌ Reranker init failed: {e}")
            engine._reranker = None

    # Configuration
    RERANK_ENABLED = True and (engine._reranker is not None)
    RERANK_FETCH_K = 30
    FINAL_TOP_K = DEFAULT_TOP_K_DYNAMIC

    # Step 1: Multi-Query Retrieval
    all_candidates = []

    # Adjust per-query budget for multi-query expansion
    if len(retrieval_queries) > 1:
        # Boost budget by 50% to account for overlap
        effective_fetch_k = int(RERANK_FETCH_K * 1.5) if RERANK_ENABLED else RERANK_FETCH_K
        k_per_query = max(10, effective_fetch_k // len(retrieval_queries))
    else:
        k_per_query = RERANK_FETCH_K if RERANK_ENABLED else DEFAULT_TOP_K_DYNAMIC

    logger.info(f"📊 Retrieval budget: {k_per_query} docs/query × {len(retrieval_queries)} queries")

    for i, query in enumerate(retrieval_queries, 1):
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"🔍 Query {i}/{len(retrieval_queries)}: {query_preview}")
        
        try:
            docs = _retrieve_docs(engine.retriever, query, k=k_per_query)
            all_candidates.extend(docs)
            logger.info(f"  ✅ {len(docs)} docs")
        
        except Exception as e:
            logger.error(f"  ❌ Error: {e}")

    # Step 2: Deduplicate
    candidate_docs = deduplicate_documents(all_candidates)
    logger.info(f"📚 Unique candidates: {len(candidate_docs)}")

    # Step 3: Filter by type/quality
    filtered_candidates = [
        d for d in candidate_docs 
        if not should_exclude_doc(d.metadata or {}, d.page_content, intent)
    ]
    logger.info(f"🔍 Filtered: {len(filtered_candidates)}/{len(candidate_docs)}")

    # Step 4: Re-rank con boost GERARCHICO delle fonti
    # MASTER FAQ > FAQ operatori > FAQ web > Guide procedurali > tutto il resto
    FAQ_BOOST = {
        "chunk_category": {
            "master_faq": 5.0,        # Sacro graal: +400% priorita'
            "faq": 2.5,               # FAQ operatori: +150%
            "web_faq": 1.8,           # FAQ web: +80%
            "guide_procedural": 1.3,  # Guide procedurali: +30%
            "access_procedure": 1.2,  # Procedure accesso: +20%
        }
    }

    # STAGE 1: Prova prima a cercare SOLO nel master FAQ
    # Se trova match forti, usa solo quelli
    master_faq_candidates = [
        d for d in filtered_candidates
        if d.metadata.get("chunk_category") == "master_faq"
    ]

    docs = None
    used_master_only = False

    if RERANK_ENABLED and master_faq_candidates:
        logger.info(f"🎯 STAGE 1: Reranking solo MASTER FAQ ({len(master_faq_candidates)} candidati)")

        try:
            master_reranked = engine._reranker.rerank(
                query=safe_q,
                documents=master_faq_candidates,
                top_k=min(FINAL_TOP_K, len(master_faq_candidates)),
                return_scores=True
            )

            # Check strong match (cross-encoder score > 0.5)
            if master_reranked and master_reranked[0][1] > 0.5:
                logger.info(f"⭐ MASTER FAQ strong match (score={master_reranked[0][1]:.2f}) → uso solo master FAQ")
                docs = [doc for doc, score in master_reranked if score > 0.3]
                if not docs:
                    docs = [doc for doc, _ in master_reranked[:FINAL_TOP_K]]
                used_master_only = True
            else:
                top_score = master_reranked[0][1] if master_reranked else 0
                logger.info(f"⚠️ MASTER FAQ weak match (score={top_score:.2f}) → fallback a retrieval completo")

        except Exception as e:
            logger.error(f"❌ Stage 1 error: {e}")

    # STAGE 2: Fallback — retrieval completo con boost master_faq
    if docs is None:
        if RERANK_ENABLED and filtered_candidates:
            logger.info(f"🔄 STAGE 2: Re-ranking completo {len(filtered_candidates)} → top-{FINAL_TOP_K}")

            try:
                docs = engine._reranker.rerank_with_metadata(
                    query=safe_q,
                    documents=filtered_candidates,
                    top_k=FINAL_TOP_K,
                    metadata_boost=FAQ_BOOST
                )

                # Log mix di fonti nel top
                cats = {}
                for d in docs:
                    cat = d.metadata.get("chunk_category", "?")
                    cats[cat] = cats.get(cat, 0) + 1
                logger.info(f"✅ Re-ranked. Distribuzione top-{FINAL_TOP_K}: {cats}")

            except Exception as e:
                logger.error(f"❌ Re-ranking error: {e} → fallback")
                docs = filtered_candidates[:FINAL_TOP_K]
        else:
            docs = filtered_candidates[:FINAL_TOP_K]

    # Log di debug
    if used_master_only:
        logger.info(f"📚 FONTE FINALE: MASTER FAQ ({len(docs)} chunk)")
    else:
        master_count = sum(1 for d in docs if d.metadata.get("chunk_category") == "master_faq")
        if master_count > 0:
            logger.info(f"📚 FONTE FINALE: Mix (master_faq={master_count}/{len(docs)})")
        else:
            logger.info(f"📚 FONTE FINALE: PDF/web (no master FAQ)")

    logger.info(f"📄 Final docs: {len(docs)}")

    # ========================================
    # 3.5 EXTRACT CONTEXTS FOR RAGAS
    # ========================================

    retrieved_contexts = [
        d.page_content[:MAX_CTX_CHARS] 
        for d in docs if hasattr(d, "page_content")
    ]

    logger.info(f"📝 Extracted {len(retrieved_contexts)} contexts")

    context_usable = is_context_usable(docs)

    if not context_usable:
        logger.warning("⚠️ Context not usable")
        return {
            "answer": "Non ho trovato informazioni pertinenti.",
            "sources": [],
            "retrieved_contexts": retrieved_contexts,
            "latency": round(time.time() - start_time, 3),
            "bm25_active": engine.bm25_active,
            "safety_flags": flags
        }

    # Canonical concepts filtering
    if canonical_q:
        canon_set = set(canonical_q)
        
        def _doc_concepts(d):
            m = getattr(d, "metadata", {}) or {}
            c = m.get("canonical_concepts") or []
            if isinstance(c, str):
                c = [x.strip() for x in c.split(",") if x.strip()]
            return set(c) if isinstance(c, list) else set()
        
        filtered = [d for d in docs if _doc_concepts(d) & canon_set]
        if filtered:
            docs = filtered
            logger.info(f"🎯 Canonical filter: {len(docs)} docs")


    # ========================================
    # 4. HYBRID STRATEGY ANSWER GENERATION
    # ========================================
    
    # Strategy A: Direct LLM per definitional/acronym
    # Filosofia: LLM comprende semantica meglio di rule-based extraction
    if intent in ["definitional", "acronym_definition"]:
        logger.info(f"📖 {intent} → LLM Direct Strategy (semantic understanding)")
        
        try:
            from core.layers.layer8_llm_strategies import llm_direct_answer
            
            answer, used_docs = llm_direct_answer(
                question=safe_q,
                docs=docs,
                llm=engine.llm,
                chat_history=chat_history
            )
            
            # Valida risposta
            if answer and len(answer) > 60 and not is_refusal(answer):
                sources = format_sources_from_docs(used_docs)
                latency = round(time.time() - start_time, 3)
                
                # Output safety check
                safe_ans, out_flags, _ = engine.safety.sanitize_input(answer)
                if out_flags.get("pii_detected"):
                    logger.warning("🔒 PII detected in output → sanitized")
                    answer = safe_ans
                
                return {
                    "answer": answer,
                    "sources": sources,
                    "retrieved_contexts": retrieved_contexts,
                    "latency": latency,
                    "bm25_active": engine.bm25_active,
                    "safety_flags": flags,
                    "answer_mode": f"llm_direct_{intent}"
                }
            
            logger.warning(f"⚠️ LLM Direct generated invalid answer → fallback")
        
        except Exception as e:
            logger.error(f"❌ LLM Direct Strategy error: {e}")
            # Fallback continua sotto
    
    # Strategy B: Procedural Synthesis per procedural
    # Filosofia: Step-by-step guidato, evidence-based quando possibile
    elif intent == "procedural":
        logger.info("🔧 Procedural Synthesis Strategy")
        
        try:
            from core.layers.layer8_llm_strategies import llm_procedural_synthesis
            
            answer, used_docs = llm_procedural_synthesis(
                question=safe_q,
                docs=docs,
                llm=engine.llm
            )
            
            # Valida risposta
            if answer and len(answer) > 60:
                sources = format_sources_from_docs(used_docs)
                latency = round(time.time() - start_time, 3)
                
                # Output safety
                safe_ans, out_flags, _ = engine.safety.sanitize_input(answer)
                if out_flags.get("pii_detected"):
                    logger.warning("🔒 PII detected → sanitized")
                    answer = safe_ans
                
                return {
                    "answer": answer,
                    "sources": sources,
                    "retrieved_contexts": retrieved_contexts,
                    "latency": latency,
                    "bm25_active": engine.bm25_active,
                    "safety_flags": flags,
                    "answer_mode": "procedural_synthesis"
                }
            
            logger.warning("⚠️ Procedural synthesis invalid → fallback")
        
        except Exception as e:
            logger.error(f"❌ Procedural Synthesis error: {e}")
            # Fallback continua sotto
    
    # ========================================
    # 4c. LLM GENERATION (strategia principale per generic e fallback)
    # ========================================
    # NOTA: Strategy C (evidence-based/atom extraction) RIMOSSA.
    # Produceva risposte con solo telefono/email ignorando il contenuto reale.
    # Il LLM e' molto piu' affidabile per generare risposte complete.
    logger.info("🤖 LLM generation (primary for generic + fallback)")
    
    system_prompt_text = get_system_prompt()
    sources = format_sources_from_docs(docs)
    links_str = format_links_for_prompt(docs)
    
    context_str_raw = format_docs_for_prompt(docs)
    context_str_enriched = context_str_raw  # No temporal enrichment (causava note spurie)

    final_prompt = FULL_PROMPT_TEMPLATE.format(
        system_prompt_text=system_prompt_text,
        context_str=context_str_enriched,
        links_str=links_str,
        chat_history_str=format_chat_history(chat_history or []),
        question=safe_q
    )

    try:
        response = engine.llm.invoke(final_prompt)
        answer = response.content.strip()

        # Se LLM rifiuta ma il contesto e' usabile, riprova con prompt piu' diretto
        if context_usable and is_refusal(answer):
            logger.warning("⚠️ LLM refusal on usable context → retry con prompt diretto")

            retry_prompt = f"""Basandoti ESCLUSIVAMENTE sui seguenti documenti, rispondi alla domanda.
NON dire "non ho trovato" — i documenti contengono informazioni utili, usale.

DOCUMENTI:
{context_str_enriched}

DOMANDA: {safe_q}

RISPOSTA (basata sui documenti sopra):"""

            try:
                retry_response = engine.llm.invoke(retry_prompt)
                retry_answer = retry_response.content.strip()
                if retry_answer and not is_refusal(retry_answer):
                    answer = retry_answer
            except Exception:
                pass  # Usa la risposta originale

        # Output safety check
        safe_ans, out_flags, _ = engine.safety.sanitize_input(answer)
        if out_flags.get("pii_detected"):
            logger.warning("🔒 PII detected in output → sanitized")
            answer = safe_ans
            
    except Exception as e:
        logger.error(f"❌ LLM generation error: {e}")
        return {
            "answer": "Errore nella generazione della risposta.", 
            "error": str(e),
            "sources": [],
            "bm25_active": engine.bm25_active,
            "retrieved_contexts": retrieved_contexts,
            "latency": round(time.time() - start_time, 3),
            "safety_flags": flags
        }

    # ========================================
    # 6. RETURN RESPONSE
    # ========================================
    latency = round(time.time() - start_time, 3)
    kpi_logger.info("rag_interaction", extra={"latency": latency, "docs": len(docs)})

    response = {
        "answer": answer,
        "sources": format_sources_from_docs(docs),
        "retrieved_contexts": retrieved_contexts,
        "bm25_active": engine.bm25_active,
        "latency": latency,
        "safety_flags": flags,
        "answer_mode": "llm_generation"
    }

    # Salva in cache
    _response_cache[cache_key] = response
    return response
# ==========================
#   CLI DI TEST
# ==========================
if __name__ == "__main__":
    print(f"\n🤖 PALES-AI RAG Engine (Layer 8) - Ready (Model: {LLM_MODEL})")
    print("Scrivi 'exit' per uscire.\n")
    
    # Warmup (opzionale, per caricare i modelli in memoria)
    print("⏳ Caricamento risorse...")
    try:
        # Inizializza il singleton
        engine = RAGEngine()
        print("✅ Motore pronto.")
    except Exception as e:
        print(f"❌ Errore avvio motore: {e}")
        sys.exit(1)

    while True:
        try:
            q = input("\nTu: ")
            if q.lower() in ["exit", "quit"]:
                break
            
            # Simuliamo una chiamata operatore
            res = process_request(q)
            
            print(f"\nPALES-AI: {res['answer']}")
            
            # Info di debug utili
            if res.get("sources"):
                print(f"\n📚 Fonti ({len(res['sources'])}):")
                for s in res['sources']:
                    print(f" - {s.get('filename')} (Pag. {s.get('page_number')})")
            
            print(f"⏱ Latency: {res.get('latency')}s")
            
            if res.get('error'):
                print(f"⚠ Error: {res['error']}")
                
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Errore loop: {e}")
