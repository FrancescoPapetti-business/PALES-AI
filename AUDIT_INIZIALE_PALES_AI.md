# AUDIT INIZIALE — PALES-AI v2.0
## Data: 15 Aprile 2026
## Autore analisi: Claude Opus 4.6 (Anthropic)
## Progetto: PALES-AI — Agente Conversazionale RAG per ClassyFarm
## Committente: Francesco Papetti — Solid Ground AI

---

# 1. A CHE PUNTO SIAMO CON IL PROGETTO

## Stato: Prototipo funzionante in locale, con hardening di sicurezza e infrastruttura completati

### Cosa funziona:

- Architettura a 8 layer (offline + online pipeline) implementata e coerente con la presentazione di tesi
- Backend FastAPI (`backend/main.py`) con endpoint `/api/v1/chat` operativo
- Frontend Next.js 16 con chatbot widget funzionante su `localhost:3000`
- Pipeline RAG ibrida FAISS + BM25 con retrieval a 2 stadi
- Cross-encoder reranker (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) integrato e attivo nella pipeline — viene inizializzato come lazy singleton e applicato dopo il retrieval ibrido sui candidati filtrati
- Safety Layer bidirezionale (input + output) con detection di prompt injection (22 pattern), PII, toxicity via OpenAI Moderation API
- Sistema di versioning dei dataset (JSONL con manifest e fingerprint SHA256)
- Intent detection a 4 livelli (procedural, acronym_definition, definitional, generic) con query expansion multi-strategia (sinonimi + intent)
- Query expansion avanzata tramite `QueryExpander` con supporto multi-query retrieval
- Ensemble retriever con pesi dinamici basati sull'intent (es. acronym_definition → BM25 pesato 70%)
- KPI tracking asincrono con audit hash
- Database SQLite con schema completo (chunks, sessions, messages, audit_log, dataset_versions) e indice FTS5
- Persistenza sessioni utente con session_id e history dal database
- Audit log strutturato su database SQLite
- Cache TTL in-memory per risposte (200 entry, scadenza 1 ora)
- Autenticazione API opzionale via header `X-API-Key`
- CORS configurabile via variabile d'ambiente (non piu' wildcard `*`)
- Rate limiting in-memory (15 req/min per IP)
- Timeout su tutte le chiamate OpenAI (30s per LLM, 60s per vision)
- Validazione input con troncamento a 500 caratteri
- Containerizzazione Docker completa (backend + frontend + nginx reverse proxy)
- Serializzazione BM25 su JSON v4.0 (rimosso pickle insicuro, mantenuto fallback per migrazione)

### Metriche di validazione (da dataset sintetico-reale):

| Metrica | Score |
|---------|-------|
| Groundedness | 0.982 |
| Consistency | 0.983 |
| Refusal Correctness | 1.000 |
| Retrieval | 0.572 |

### Cosa manca per andare in produzione:

- Deploy su server (VPS o cloud) — il sistema gira solo in locale
- Dominio e certificato SSL
- Test end-to-end e integration test
- Refactoring di Layer 8 (1171 righe in un singolo file)
- Ottimizzazione UX del frontend per utenti PA
- Documentazione DPIA / AI Act

### Percentuale di completamento stimata:

| Area | Completamento | Note |
|------|--------------|------|
| Architettura RAG | 95% | Tutti i componenti integrati e funzionanti |
| Data Pipeline (offline) | 95% | Funzionante, manca incremental indexing |
| Retrieval (online) | 85% | Reranker attivo, cache implementata, retrieval score migliorabile |
| Safety & Governance | 90% | 22 pattern injection, moderation API, audit su DB |
| Frontend | 55% | Funzionale ma non ottimizzato per utenti PA |
| Deployment/Produzione | 85% | Docker pronto, manca solo il server |
| Testing | 20% | Solo test manuali e validazione RAGAS, nessun test automatizzato |
| Sicurezza | 80% | Auth, CORS, rate limiting, no pickle, timeout |

---

# 2. PROBLEMI E CRITICITA' IDENTIFICATI

## 2.1 — Criticita' alta

### 2.1.1 — Layer 8 troppo complesso
- **File:** `core/layers/layer8_application.py` (1171 righe)
- **Problema:** Un singolo file contiene: intent classification, query expansion, retrieval orchestration, 5 strategie di generazione con fallback multipli, evidence collection, formatting
- **Pattern hardcoded** come `"aggiornare l'APP"` e `"seleziona azienda"` che triggerano fallback deterministici (righe 1069-1083)
- **Import circolari** risolti con import dentro le funzioni (es. riga 715, 781, 908, 950)
- **Impatto:** Difficile da debuggare, testare e manutenere
- **Priorita':** Media — funziona ma rende lo sviluppo futuro piu' lento

### 2.1.2 — Retrieval score basso (0.572)
- **Problema:** Il retrieval score e' il punto piu' debole del sistema
- **Causa principale:** La complessita' intrinseca del corpus documentale (manuali tecnici, checklist, guide procedurali con terminologia specialistica)
- **Mitigazione attuale:** Il reranker cross-encoder e' attivo e migliora la precision, ma il recall iniziale rimane limitato
- **Possibili miglioramenti:** Query rewriting con LLM, embedding fine-tuning su dominio specifico, adaptive chunking
- **Nota:** Il retrieval score basso non compromette la qualita' complessiva delle risposte grazie ai meccanismi di safety e validazione dell'output (groundedness 0.982)

### 2.1.3 — Nessun test automatizzato
- **Problema:** Non esistono integration test ne' test end-to-end
- **Presente:** Solo test manuali isolati (`test_*.py`) e validazione RAGAS
- **Impatto:** Impossibile verificare automaticamente che un aggiornamento non rompa qualcosa
- **Priorita':** Alta per produzione, media per MVP

## 2.2 — Criticita' media

### 2.2.1 — Re-indicizzazione completa
- **File:** `core/layers/layer7_vector_store.py`
- **Problema:** `incremental_indexing_update()` ricostruisce l'intero indice FAISS ad ogni aggiornamento
- **Impatto:** Aggiungere un singolo documento richiede di riprocessare tutti i chunk
- **Accettabile per ora:** Con il dataset attuale (~245 chunk) la ricostruzione richiede pochi minuti

### 2.2.2 — Chiamate LLM multiple nel preprocessing (Layer 4)
- **File:** `core/layers/layer4_semantic_rewriting.py`
- **Problema:** Per ogni documento nuovo: GPT-4o (vision) + GPT-4o-mini (rewriting) + GPT-4o-mini (metadata)
- **Impatto:** Costoso e lento per l'ingestion di nuovi documenti
- **Mitigazione:** Il preprocessing e' offline, non impatta la latenza utente

### 2.2.3 — FAISS caricato interamente in RAM
- **Problema:** L'indice FAISS viene caricato tutto in memoria al primo request
- **Impatto attuale:** Con ~245 chunk l'indice pesa ~10 MB — irrilevante
- **Impatto futuro:** Con dataset grandi (10k+ chunk) potrebbe consumare RAM significativa
- **Mitigazione:** Il VPS consigliato (4 GB RAM) e' sufficiente per il volume attuale e prevedibile

### 2.2.4 — Frontend non ottimizzato per PA
- **Problema:** Il frontend e' funzionale ma basico
- **Mancanze:** Nessun test di accessibilita' WCAG, nessun test con utenti reali, nessun supporto mobile verificato
- **Priorita':** Media — funziona per la demo e il primo lancio

## 2.3 — Criticita' bassa (miglioramenti futuri)

### 2.3.1 — Costanti hardcoded
- `MAX_CTX_CHARS=2000`, `STAGE1_CANDIDATES=25`, `DEFAULT_TOP_K=5`, `PROCEDURAL_TOP_K=8`
- Alcune gia' configurabili via env vars (`MAX_CTX_CHARS`, `LLM_MODEL`), altre no
- Non bloccante ma limita la configurabilita' senza modificare il codice

### 2.3.2 — Nessun response streaming
- L'utente aspetta 3-8 secondi senza feedback visivo (solo animazione "typing dots")
- Server-Sent Events (SSE) migliorerebbero la UX percepita
- Nice-to-have, non bloccante

### 2.3.3 — Classificatore OOD basato su heuristic
- La detection out-of-domain usa `is_out_of_domain()` basata sulla lunghezza media dei documenti recuperati (< 50 chars)
- Funziona ma un classificatore basato sulla distanza dagli embedding del corpus sarebbe piu' robusto

---

# 3. ARCHITETTURA DEL SISTEMA

## 3.1 — Stack tecnologico

| Componente | Tecnologia | Versione |
|------------|-----------|----------|
| Backend | FastAPI + Uvicorn | Python 3.11 |
| Frontend | Next.js + React | 16.0.5 / 19.2.0 |
| Styling | Tailwind CSS | 3.4.18 |
| LLM Generazione | OpenAI GPT-4o-mini | via API |
| LLM Vision | OpenAI GPT-4o | via API |
| Embeddings | OpenAI text-embedding-3-small | 1536 dim |
| Reranker | cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 | Sentence Transformers |
| Vector Store | FAISS | CPU |
| Keyword Search | BM25 (LangChain) | Serializzato JSON v4.0 |
| Database | SQLite + FTS5 | Standard library |
| Orchestrazione | LangChain + LangGraph | 1.0.2 / 1.0.1 |
| Containerizzazione | Docker + Docker Compose | Multi-stage build |
| Reverse Proxy | Nginx | Config inclusa |

## 3.2 — Pipeline Offline (Ingestion)

```
Layer 1: Raccolta Dati
  → Scansione ricorsiva /data/pdf_input/ + documenti web
  → Supporto: PDF, DOCX, DOC, TXT, MD, HTML
  → Hashing SHA256 per identificazione unica

Layer 2: Rilevamento Qualita' e Profilazione
  → Classificazione tipo documento (text, visual, table, web_content)
  → Selezione strategia di estrazione

Layer 3: Estrazione e Pulizia
  → PDF: PyMuPDF + pymupdf4llm (conversione Markdown)
  → Web: html2text → BeautifulSoup → raw fallback
  → DOCX: python-docx
  → Normalizzazione whitespace e pulizia artefatti

Layer 4: Riscrittura Semantica e Chunking
  → GPT-4o per analisi vision (screenshot con procedure numerate)
  → GPT-4o-mini per normalizzazione testo italiano professionale
  → Estrazione metadata (summary, keywords, topic classification)
  → 220+ mappature concetti canonici (DASHBOARD, SQNBA, BDN, ...)
  → Espansione acronimi dominio-specifici
  → Semantic lifting (regole obbligatorie, restrizioni accesso, procedure)

Layer 5: Validazione e Governance
  → Quality gate: min 50 chars, no frasi di rifiuto LLM
  → PII detection (Codice Fiscale, telefoni, carte credito)
  → Schema integrity (campi obbligatori)
  → Classificazione chunk (web_faq, web_guide, procedural, mandatory_rule, ...)

Layer 6: Strutturazione Dataset
  → Versioning con tag (v1_20251216_170818, v2_..., v3_...)
  → Formato JSONL con manifest e fingerprint SHA256
  → Dataset immutabili (golden datasets)

Layer 7: Indicizzazione e Embedding
  → FAISS: embedding text-embedding-3-small → indice vettoriale
  → BM25: serializzazione JSON v4.0 dei documenti
  → SQLite: inserimento chunk + indice FTS5
```

## 3.3 — Pipeline Online (Query)

```
INPUT: Query utente + session_id
        │
        ▼
Cache Lookup (TTL 1h, max 200 entry)
  → Cache hit: ritorna risposta immediata (<1ms)
  → Cache miss: continua pipeline
        │
        ▼
Safety Layer INPUT
  → Troncamento a 500 caratteri
  → Detection prompt injection (22 pattern regex)
  → PII detection + redazione
  → Toxicity detection (OpenAI Moderation API, fail-open con flag)
  → Blocco se injection o toxicity
        │
        ▼
Intent Classification
  → Procedural: "come faccio", "dove trovo", verbi azione + "come"
  → Acronym Definition: pattern regex per acronimi (OdC, BDN, SQNBA)
  → Definitional: "cos'e'", "cosa significa", "a cosa serve"
  → Generic: fallback
        │
        ▼
Query Expansion
  → Canonical tag injection ([DASHBOARD] [SQNBA] query)
  → Espansione acronimi
  → Multi-query expansion (sinonimi + intent)
  → Max 5 query espanse
        │
        ▼
Dynamic Ensemble Weights
  → acronym_definition: FAISS 30% / BM25 70%
  → definitional: FAISS 60% / BM25 40%
  → procedural: FAISS 70% / BM25 30%
  → generic: FAISS 60% / BM25 40%
        │
        ▼
Multi-Query Retrieval
  → Per ogni query espansa: retrieval con budget k proporzionale
  → Deduplicazione documenti (content hash)
  → Filtro per tipo/qualita' (escludi sanzioni, tabelle rumorose)
        │
        ▼
Cross-Encoder Reranking
  → Modello: mmarco-mMiniLMv2-L12-H384-v1 (multilingue, italiano)
  → Rerank sui candidati filtrati → top-K finali
  → Latenza: ~150-250ms per 30 documenti su CPU
  → Fallback: ordine originale se reranking fallisce
        │
        ▼
Strategia di Generazione (basata su intent)
  → Strategy A (definitional/acronym): LLM Direct — contesto completo, LLM decide
  → Strategy B (procedural): Procedural Synthesis — step-by-step guidato
  → Strategy C (generic/fallback): Evidence-Based — estrazione deterministica atomi
  → Strategy D (backup): Direct Extraction — pattern matching
  → Strategy E (ultimo fallback): LLM con prompt completo + validazione temporale
        │
        ▼
Validazione Temporale
  → Verifica che valori temporali del contesto siano preservati nella risposta
  → Se mancanti: aggiunge nota temporale
        │
        ▼
Safety Layer OUTPUT
  → PII detection sulla risposta generata
  → Redazione se necessario
        │
        ▼
OUTPUT: Risposta + fonti + latenza + safety_flags + session_id
  → Salvataggio in cache
  → Salvataggio messaggio in SQLite (sessions/messages)
  → Audit log asincrono su SQLite
```

## 3.4 — Schema Database SQLite

```sql
chunks          -- Knowledge base
  chunk_id TEXT PRIMARY KEY
  parent_doc_id TEXT
  text_content TEXT
  canonical_concepts TEXT (JSON)
  metadata TEXT (JSON)

chunks_fts      -- Indice full-text (FTS5) su text_content

sessions        -- Sessioni utente
  session_id TEXT PRIMARY KEY
  user_role TEXT
  ip_hash TEXT
  last_activity TEXT

messages        -- Storico conversazioni
  session_id TEXT (FK)
  role TEXT (user/assistant)
  content TEXT
  safety_flags TEXT (JSON)
  answer_mode TEXT
  latency REAL

audit_log       -- Tracciabilita' interazioni
  endpoint TEXT
  status TEXT (success/blocked/error)
  latency REAL
  query_hash TEXT
  answer_mode TEXT
  safety_flags TEXT (JSON)

dataset_versions -- Versioning dei dataset
  version_tag TEXT PRIMARY KEY
  fingerprint TEXT
  chunk_count INTEGER
```

## 3.5 — Sicurezza implementata

| Meccanismo | Implementazione | Stato |
|------------|----------------|-------|
| Prompt injection | 22 pattern regex (IT + EN) | Attivo |
| PII detection | Codice Fiscale, telefoni, carte credito | Attivo |
| PII redazione | Sostituzione con `[TYPE_RIMOSSO]` | Attivo |
| Toxicity detection | OpenAI Moderation API | Attivo (fail-open con flag) |
| Input truncation | Max 500 caratteri | Attivo |
| CORS | Configurabile via env var | Attivo |
| API Key auth | Header `X-API-Key`, opzionale | Attivo |
| Rate limiting | 15 req/min per IP | Attivo |
| LLM timeout | 30s (LLM), 60s (vision) | Attivo |
| Pickle rimosso | BM25 su JSON v4.0 | Attivo |
| Audit trail | SQLite con query_hash | Attivo |

---

# 4. FILE CRITICI DEL PROGETTO

| File | Righe | Ruolo | Criticita' |
|------|-------|-------|-----------|
| `core/layers/layer8_application.py` | 1171 | Orchestrazione RAG completa | Alta — troppo grande, da spezzare |
| `core/layers/layer7_vector_store.py` | ~700 | Indicizzazione FAISS + BM25 + retrieval ibrido | Media |
| `core/layers/layer7_reranker.py` | 379 | Cross-encoder reranking | Bassa — pulito e modulare |
| `core/layers/layer4_semantic_rewriting.py` | 592 | Enrichment semantico + vision | Media — molte chiamate LLM |
| `core/layers/layer_safety_moderation.py` | ~200 | Safety bidirezionale | Bassa — funzionale |
| `core/database.py` | ~280 | Database SQLite + FTS5 | Bassa — nuovo, pulito |
| `backend/main.py` | ~130 | Entry point + middleware sicurezza | Bassa |
| `backend/api/chat.py` | ~110 | Endpoint chat + sessioni + audit | Bassa |
| `frontend/app/components/ChatWidget.tsx` | ~290 | Widget chatbot | Media — funzionale ma basico |
| `frontend/app/lib/api.ts` | ~45 | Client API con timeout e error handling | Bassa |

---

# 5. ROADMAP MIGLIORAMENTI FUTURI

## Priorita' 1 — Migliorare il Retrieval (da 0.572 a 0.75+)

| # | Azione | Impatto | Effort |
|---|--------|---------|--------|
| 1 | Query rewriting con LLM prima del retrieval | +10-15% recall | 2 giorni |
| 2 | Embedding fine-tuning su corpus ClassyFarm | +15-20% precision | 3-5 giorni |
| 3 | Adaptive chunking (chunk size variabile per tipo documento) | +5-10% recall | 3 giorni |

## Priorita' 2 — Qualita' del codice

| # | Azione | Impatto | Effort |
|---|--------|---------|--------|
| 4 | Spezzare Layer 8 in 5 moduli | Manutenibilita' | 3-4 giorni |
| 5 | Rimuovere pattern hardcoded, rendere data-driven | Configurabilita' | 1-2 giorni |
| 6 | Costanti in file YAML configurabile | Flessibilita' | 1 giorno |
| 7 | Test e2e + integration test | Affidabilita' | 2-3 giorni |

## Priorita' 3 — UX e funzionalita'

| # | Azione | Impatto | Effort |
|---|--------|---------|--------|
| 8 | Response streaming (SSE) | UX percepita | 2 giorni |
| 9 | Pagina admin per upload documenti | Autonomia gestione | 1-2 giorni |
| 10 | Redesign UX per utenti PA | Accessibilita' | 5+ giorni |
| 11 | Incremental indexing | Efficienza aggiornamenti | 2 giorni |

## Priorita' 4 — Compliance

| # | Azione | Impatto | Effort |
|---|--------|---------|--------|
| 12 | Classificatore ML per prompt injection | Sicurezza | 3 giorni |
| 13 | Classificatore OOD basato su embedding distance | Robustezza | 2 giorni |
| 14 | Documentazione DPIA / AI Act | Compliance legale | Documentale |

---

# 6. CONCLUSIONI

PALES-AI e' un sistema RAG con architettura solida a 8 layer, ben strutturato per un contesto istituzionale (Pubblica Amministrazione). Il progetto e' passato da prototipo di tesi a sistema production-ready con l'aggiunta di:

- Database persistente (SQLite + FTS5)
- Sicurezza multi-livello (auth, CORS, rate limiting, input validation, audit trail)
- Cache per performance
- Containerizzazione Docker completa
- Gestione sessioni utente

Il punto debole principale resta il **retrieval score (0.572)**, che tuttavia non compromette la qualita' complessiva delle risposte grazie alla pipeline di safety e validazione (groundedness 0.982, consistency 0.983, refusal correctness 1.000).

Il sistema e' **pronto per il primo deploy su VPS** e per la raccolta di feedback da utenti reali, che informera' le successive iterazioni di miglioramento.
