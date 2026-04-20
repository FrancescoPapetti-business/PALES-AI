"""
Script di ricostruzione completa della Knowledge Base PALES-AI.

Cosa fa:
1. Processa TUTTI i PDF in data/pdf_input/ (che erano ignorati)
2. Processa le pagine web HTML gia' presenti
3. Filtra chunk spazzatura (cookie, nav, header, footer)
4. Deduplica chunk per contenuto
5. Calcola word_count per ogni chunk
6. Salva un dataset JSONL pulito
7. Ricostruisce gli indici FAISS + BM25 + SQLite

Uso: python -m scripts.rebuild_knowledge_base
"""

import sys
import os
import json
import hashlib
import uuid
import re
from pathlib import Path
from datetime import datetime

# Setup path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

# ==================================
# CONFIGURAZIONE
# ==================================

PDF_INPUT_DIR = BASE_DIR / "data" / "pdf_input"
FAQ_INPUT_DIR = BASE_DIR / "data" / "faq_input"
MASTER_FAQ_DIR = BASE_DIR / "data" / "master_faq"
WEB_DATASET_DIR = BASE_DIR / "data" / "datasets"
OUTPUT_DIR = BASE_DIR / "data" / "datasets"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

# Chunking parameters
CHUNK_SIZE = 2000       # caratteri (era 1000, troppo piccolo)
CHUNK_OVERLAP = 300     # caratteri di sovrapposizione
MIN_CHUNK_WORDS = 30    # chunk con meno parole vengono scartati
MAX_CHUNK_WORDS = 800   # chunk piu' lunghi vengono ri-spezzati

# Pattern di contenuto spazzatura da filtrare
JUNK_PATTERNS = [
    r"seleziona la tua lingua",
    r"ho compreso",
    r"cookie\s*(policy|banner|consent)",
    r"questo sito utilizza cookie",
    r"accetta\s*(tutti|cookie)",
    r"informativa\s*(sulla\s*)?privacy",
    r"^\s*\[?\s*(DE|EN|FR|ES)\s*\]?\s*$",
    r"^\s*menu\s*$",
    r"^\s*home\s*$",
    r"^\s*contatti\s*$",
    r"^\s*cerca\s*$",
    r"^\s*accedi\s*$",
    r"^\s*(copyright|©)\s*\d{4}",
    r"tutti i diritti riservati",
    r"^\s*telefono\s*:\s*\d",
    r"^\s*email\s*:\s*\S+@\S+\s*$",
    r"^\s*fax\s*:\s*\d",
    r"seguici su",
    r"iscriviti alla newsletter",
    r"condividi\s*(su|con)",
    r"stampa questa pagina",
    r"torna su",
    r"torna all.inizio",
    r"skip to content",
    r"vai al contenuto",
]

JUNK_COMPILED = [re.compile(p, re.IGNORECASE) for p in JUNK_PATTERNS]


# ==================================
# 1. ESTRAZIONE PDF
# ==================================

def extract_pdf_pages(pdf_path: Path) -> list:
    """Estrae testo pagina per pagina da un PDF usando PyMuPDF."""
    try:
        import fitz  # pymupdf
    except ImportError:
        print("  ERRORE: pymupdf non installato. Esegui: pip install pymupdf")
        return []

    pages = []
    try:
        doc = fitz.open(str(pdf_path))
        for page_num, page in enumerate(doc, 1):
            text = page.get_text("text")
            if text and text.strip():
                pages.append({
                    "text": text.strip(),
                    "page_number": page_num,
                    "filename": pdf_path.name
                })
        doc.close()
    except Exception as e:
        print(f"  ERRORE PDF {pdf_path.name}: {e}")

    return pages


def collect_all_pdfs() -> list:
    """Raccoglie tutti i PDF da data/pdf_input/ ricorsivamente."""
    pdfs = []
    for pdf_path in PDF_INPUT_DIR.rglob("*.pdf"):
        pdfs.append(pdf_path)
    for pdf_path in PDF_INPUT_DIR.rglob("*.PDF"):
        if pdf_path not in pdfs:
            pdfs.append(pdf_path)
    return sorted(pdfs)


# ==================================
# 2. ESTRAZIONE WEB (dal dataset esistente)
# ==================================

def load_existing_web_chunks() -> list:
    """Carica chunk web dal dataset esistente (se utili)."""
    chunks = []
    # Cerca l'ultimo dataset
    versions = sorted(WEB_DATASET_DIR.glob("v*"), reverse=True)

    for v in versions:
        chunks_file = v / "chunks.jsonl"
        if chunks_file.exists():
            with open(chunks_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        # Includi solo chunk web con contenuto reale
                        text = chunk.get("text_content", "")
                        if text and len(text.split()) >= MIN_CHUNK_WORDS:
                            chunks.append(chunk)
                    except json.JSONDecodeError:
                        continue
            print(f"  Caricati {len(chunks)} chunk web da {v.name}")
            break

    return chunks


# ==================================
# 3. CHUNKING INTELLIGENTE
# ==================================

def smart_chunk_text(text: str, filename: str, page_number: int = None,
                     source_type: str = "pdf") -> list:
    """
    Chunking intelligente:
    - Prova a spezzare per paragrafi/sezioni (doppio a-capo)
    - Fallback su RecursiveCharacterTextSplitter
    - Ogni chunk ha word_count calcolato
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Pulisci il testo
    text = clean_text(text)

    if not text or len(text.split()) < MIN_CHUNK_WORDS:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " "],
        length_function=len,
    )

    docs = splitter.create_documents([text])
    chunks = []

    for i, doc in enumerate(docs):
        chunk_text = doc.page_content.strip()

        # Tronca all'ultima frase completa per evitare frasi a meta'
        chunk_text = trim_to_sentence_boundary(chunk_text)

        word_count = len(chunk_text.split())

        # Scarta chunk troppo corti
        if word_count < MIN_CHUNK_WORDS:
            continue

        # Scarta chunk spazzatura
        if is_junk(chunk_text):
            continue

        chunk_id = str(uuid.uuid4())
        parent_doc_id = hashlib.sha256(filename.encode()).hexdigest()[:16]

        chunks.append({
            "chunk_id": chunk_id,
            "parent_doc_id": parent_doc_id,
            "text_content": chunk_text,
            "canonical_concepts": extract_concepts(chunk_text),
            "metadata": {
                "filename": filename,
                "page_number": page_number,
                "source_type": source_type,
                "semantic_role": classify_role(chunk_text),
                "chunk_category": classify_category(chunk_text, source_type, filename),
                "word_count": word_count,
                "chunk_index": i,
                "summary": chunk_text[:150].replace("\n", " "),
                "keywords": extract_keywords(chunk_text),
                "extracted_links": [],
                "normalized_text": chunk_text.lower().strip(),
            }
        })

    return chunks


def clean_text(text: str) -> str:
    """Pulizia testo avanzata."""
    # Rimuovi caratteri di controllo
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # Comprimi righe vuote consecutive (max 1)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    # Comprimi spazi multipli
    text = re.sub(r' {3,}', ' ', text)
    # Rimuovi righe con solo numeri (numeri pagina)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    # Rimuovi righe con solo "PAG" o "Pag." (intestazioni pagina)
    text = re.sub(r'^\s*PAG\.?\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    # Rimuovi righe con solo trattini/underscore (separatori)
    text = re.sub(r'^[\s_\-=]{5,}$', '', text, flags=re.MULTILINE)
    # Rimuovi tab multipli (tabelle rotte)
    text = re.sub(r'\t{2,}', ' ', text)
    return text.strip()


def trim_to_sentence_boundary(text: str) -> str:
    """Tronca il testo all'ultima frase completa (finisce con . ? ! :)"""
    if not text:
        return text
    # Se finisce gia' con punteggiatura, ok
    if text.rstrip()[-1] in '.?!:':
        return text
    # Cerca l'ultimo punto/punteggiatura
    last_period = max(text.rfind('. '), text.rfind('.\n'), text.rfind('? '), text.rfind('! '), text.rfind(':\n'))
    if last_period > len(text) * 0.5:  # Solo se non perdiamo piu' del 50%
        return text[:last_period + 1]
    return text


def is_junk(text: str) -> bool:
    """Verifica se il chunk e' spazzatura."""
    t = text.lower().strip()

    # Troppo corto
    if len(t) < 20:
        return True

    # Matcha un pattern di spazzatura
    for pattern in JUNK_COMPILED:
        if pattern.search(t):
            # Se il chunk e' SOLO il pattern (poche parole), scarta
            if len(t.split()) < 15:
                return True

    # Chunk che e' solo una lista di link
    link_count = t.count("http://") + t.count("https://") + t.count(".pdf")
    word_count = len(t.split())
    if link_count > 0 and word_count < 20:
        return True

    # Chunk con troppi simboli rispetto al testo
    alpha_chars = sum(1 for c in t if c.isalpha())
    if len(t) > 0 and alpha_chars / len(t) < 0.4:
        return True

    return False


# ==================================
# 4. CLASSIFICAZIONE E METADATA
# ==================================

def classify_role(text: str) -> str:
    """Classifica il ruolo semantico del chunk."""
    t = text.lower()

    procedural_signals = [
        "clicca", "seleziona", "inserisci", "compila", "accedi",
        "step", "passo", "passaggio", "procedura", "istruzioni",
        "1.", "2.", "3.", "a)", "b)", "c)",
        "fare clic", "premere", "digitare", "scegliere",
        "dalla sezione", "nella schermata", "nel menu",
    ]
    if sum(1 for s in procedural_signals if s in t) >= 2:
        return "procedural"

    rule_signals = [
        "deve", "obbligatorio", "necessario", "tenuto a",
        "ai sensi", "in conformita'", "secondo il regolamento",
        "e' vietato", "non e' consentito", "sanzione",
    ]
    if sum(1 for s in rule_signals if s in t) >= 2:
        return "mandatory_rule"

    definition_signals = [
        "si intende", "e' definito", "con il termine",
        "si riferisce a", "indica", "rappresenta",
    ]
    if any(s in t for s in definition_signals):
        return "definitional"

    return "descriptive"


def classify_category(text: str, source_type: str, filename: str = "") -> str:
    """Classifica la categoria del chunk usando testo + filename."""
    t = text.lower()
    fn = filename.lower()

    if source_type == "web":
        if "?" in t[:100]:
            return "web_faq"
        return "web_content"

    # --- Da filename (piu' affidabile del contenuto) ---

    # Guide e manuali
    if any(kw in fn for kw in ["guida", "manuale", "manual", "man_"]):
        # Distingui guide procedurali vs manuali descrittivi
        if any(kw in fn for kw in ["inserimento", "registrazione", "accesso", "richiesta"]):
            return "guide_procedural"
        return "guide_reference"

    # Checklist
    if any(kw in fn for kw in ["check", "checklist", "check-list", "check_list"]):
        return "checklist"

    # Moduli e form
    if any(kw in fn for kw in ["modulo", "modello", "richiesta", "delega"]):
        return "form_module"

    # Circolari e normativa
    if any(kw in fn for kw in ["circolare", "decreto", "regolamento", "normativa"]):
        return "normative"

    # --- Da contenuto ---

    # Checklist dal contenuto (tabelle SI/NO, punteggi)
    si_no_count = t.count(" si ") + t.count(" no ") + t.count("si\n") + t.count("no\n")
    if si_no_count >= 3:
        return "checklist"

    # Procedure dal contenuto
    step_signals = ["passo ", "step ", "1.", "2.", "3.", "clicca", "seleziona", "inserisci"]
    if sum(1 for s in step_signals if s in t) >= 3:
        return "guide_procedural"

    # Sanzioni
    if "sanzione" in t or "infrazione" in t or "penalita'" in t:
        return "sanctions"

    # Accesso e registrazione
    if any(kw in t[:200] for kw in ["registr", "accesso al sistema", "accedi", "login"]):
        return "access_procedure"

    # Definizioni
    if any(kw in t[:200] for kw in ["si intende", "e' definito", "con il termine"]):
        return "definition"

    return "descriptive"


# Mappa concetti canonici (semplificata dal layer4)
CANONICAL_MAP = {
    "DASHBOARD": ["cruscotto", "dashboard", "pannello", "schermata principale"],
    "REGISTRAZIONE": ["registrazione", "registrare", "registrarsi", "iscrizione"],
    "ACCESSO": ["accesso", "login", "accedere", "autenticazione", "spid", "cie"],
    "DELEGA": ["delega", "delegato", "delegare", "sostituzione"],
    "CHECKLIST": ["checklist", "check-list", "check list", "lista di controllo"],
    "BENESSERE_ANIMALE": ["benessere animale", "benessere", "welfare"],
    "BIOSICUREZZA": ["biosicurezza", "bio-sicurezza", "biocheck"],
    "FARMACOSORVEGLIANZA": ["farmacosorveglianza", "farmaco", "antimicrobico", "antibiotico"],
    "VETERINARIO": ["veterinario", "veterinari", "medico veterinario"],
    "OPERATORE": ["operatore", "allevatore", "operatori"],
    "CLASSYFARM": ["classyfarm", "classy farm", "sistema classyfarm"],
    "BDN": ["bdn", "banca dati nazionale"],
    "SQNBA": ["sqnba", "sistema qualita'", "sistema qualità", "qualita' nazionale"],
    "OdC": ["odc", "organismo di certificazione", "organismo certificazione"],
    "ASL": ["asl", "azienda sanitaria", "servizio veterinario"],
    "PIANO_RIENTRO": ["piano di rientro", "piano rientro", "pdr"],
    "QUESTIONARI": ["questionari", "questionario", "sistema questionari"],
}


def extract_concepts(text: str) -> list:
    """Estrae concetti canonici dal testo."""
    t = text.lower()
    found = []
    for concept, synonyms in CANONICAL_MAP.items():
        if any(syn in t for syn in synonyms):
            found.append(concept)
    return found


def extract_keywords(text: str) -> list:
    """Estrae keyword semplici dal testo."""
    t = text.lower()
    keywords = []

    domain_terms = [
        "classyfarm", "registrazione", "accesso", "delega", "checklist",
        "benessere", "biosicurezza", "farmacosorveglianza", "veterinario",
        "operatore", "allevamento", "suini", "bovini", "avicoli", "ovicaprini",
        "conigli", "questionario", "piano di rientro", "spid", "cie",
        "bdn", "sqnba", "dashboard",
    ]

    for term in domain_terms:
        if term in t:
            keywords.append(term)

    return keywords[:10]


# ==================================
# 5. DEDUPLICAZIONE
# ==================================

def deduplicate_chunks(chunks: list) -> list:
    """Deduplica chunk basandosi su hash del contenuto normalizzato."""
    seen = set()
    unique = []

    for chunk in chunks:
        # Normalizza: lowercase, rimuovi whitespace extra
        normalized = re.sub(r'\s+', ' ', chunk["text_content"].lower().strip())
        # Hash dei primi 500 caratteri (sufficienti per identificare duplicati)
        content_hash = hashlib.md5(normalized[:500].encode()).hexdigest()

        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(chunk)

    removed = len(chunks) - len(unique)
    if removed > 0:
        print(f"  Rimossi {removed} duplicati ({len(unique)} chunk unici)")

    return unique


# ==================================
# 6. SALVATAGGIO E INDICIZZAZIONE
# ==================================

def save_dataset(chunks: list) -> Path:
    """Salva il dataset come JSONL versionato."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Trova la prossima versione
    existing = sorted(OUTPUT_DIR.glob("v*"))
    next_version = len(existing) + 1
    version_tag = f"v{next_version}_{timestamp}"

    output_path = OUTPUT_DIR / version_tag
    output_path.mkdir(parents=True, exist_ok=True)

    chunks_file = output_path / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Manifest
    manifest = {
        "version_tag": version_tag,
        "created_at": datetime.now().isoformat(),
        "chunk_count": len(chunks),
        "note": "Ricostruzione completa: PDF + web, filtrato e deduplicato",
        "fingerprint": hashlib.sha256(
            open(chunks_file, "rb").read()
        ).hexdigest()[:16],
        "sources": {
            "pdf_count": sum(1 for c in chunks if c["metadata"].get("source_type") == "pdf"),
            "web_count": sum(1 for c in chunks if c["metadata"].get("source_type") == "web"),
        }
    }

    with open(output_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n  Dataset salvato: {output_path}")
    print(f"  Manifest: {json.dumps(manifest, indent=2)}")

    return chunks_file


def rebuild_indices(chunks_file: Path):
    """Ricostruisce FAISS, BM25 e SQLite."""
    from langchain_core.documents import Document
    from core.layers.layer7_vector_store import create_bm25_index
    from core.database import init_database, insert_chunks

    # Carica chunk dal JSONL
    chunks_data = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunks_data.append(json.loads(line.strip()))

    # Converti in LangChain Documents per FAISS e BM25
    documents = []
    for chunk in chunks_data:
        doc = Document(
            page_content=chunk["text_content"],
            metadata=chunk.get("metadata", {})
        )
        # Aggiungi chunk_id al metadata per tracking
        doc.metadata["chunk_id"] = chunk["chunk_id"]
        documents.append(doc)

    print(f"\n  Documenti per indicizzazione: {len(documents)}")

    # --- FAISS ---
    print("\n  Ricostruzione FAISS...")
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    faiss_dir = VECTOR_STORE_DIR / "faiss_index"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    # Batch embedding per evitare timeout
    batch_size = 50
    print(f"  Embedding in batch di {batch_size}...")

    vectorstore = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"    Batch {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} ({len(batch)} docs)")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            batch_store = FAISS.from_documents(batch, embeddings)
            vectorstore.merge_from(batch_store)

    vectorstore.save_local(str(faiss_dir))
    print(f"  FAISS salvato: {faiss_dir}")

    # --- BM25 ---
    print("\n  Ricostruzione BM25 (JSON v4.0)...")
    create_bm25_index(documents)

    # --- SQLite ---
    print("\n  Aggiornamento SQLite + FTS5...")
    init_database()
    # Batch insert
    for i in range(0, len(chunks_data), 100):
        batch = chunks_data[i:i + 100]
        insert_chunks(batch)
        print(f"    SQLite: {min(i + 100, len(chunks_data))}/{len(chunks_data)}")

    print("\n  Tutti gli indici ricostruiti.")


# ==================================
# 7. PROCESSAMENTO FAQ DOCX
# ==================================

def process_faq_docx(docx_path: Path) -> list:
    """
    Processa il file DOMANDE FREQUENTI.docx.
    Ogni coppia domanda/risposta diventa un chunk FAQ prioritario.
    """
    try:
        from docx import Document as DocxDocument
    except ImportError:
        print("  ERRORE: python-docx non installato. Esegui: pip install python-docx")
        return []

    if not docx_path.exists():
        print(f"  File FAQ non trovato: {docx_path}")
        return []

    doc = DocxDocument(str(docx_path))
    chunks = []

    # Parsing: alterna domande (in grassetto o con ?) e risposte
    current_section = "GENERALE"
    current_question = None
    current_answer_lines = []

    def flush_qa():
        """Salva la coppia Q/A corrente come chunk."""
        nonlocal current_question, current_answer_lines
        if current_question and current_answer_lines:
            answer = " ".join(current_answer_lines).strip()
            if answer:
                text = f"DOMANDA: {current_question}\nRISPOSTA: {answer}"
                chunk_id = str(uuid.uuid4())

                role = "procedural" if any(kw in current_question.lower()
                    for kw in ["come", "dove", "quando", "quanto"]) else "definitional"

                chunks.append({
                    "chunk_id": chunk_id,
                    "parent_doc_id": hashlib.sha256(b"DOMANDE_FREQUENTI").hexdigest()[:16],
                    "text_content": text,
                    "canonical_concepts": extract_concepts(text),
                    "metadata": {
                        "filename": "DOMANDE FREQUENTI.docx",
                        "page_number": None,
                        "source_type": "faq",
                        "semantic_role": role,
                        "chunk_category": "faq",
                        "word_count": len(text.split()),
                        "chunk_index": len(chunks),
                        "summary": current_question,
                        "keywords": extract_keywords(text),
                        "extracted_links": [],
                        "faq_section": current_section,
                    }
                })
        current_question = None
        current_answer_lines = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Sezioni principali (OPERATORE E DELEGATI, ALS, VETERINARI...)
        if text.isupper() and len(text.split()) <= 5 and "?" not in text:
            flush_qa()
            current_section = text
            continue

        # Domanda (contiene ?)
        if "?" in text:
            flush_qa()
            current_question = text
            current_answer_lines = []
        elif current_question:
            # Riga di risposta
            current_answer_lines.append(text)

    # Flush ultima Q/A
    flush_qa()

    print(f"  FAQ: {len(chunks)} coppie domanda/risposta processate")
    return chunks


def process_master_faq() -> list:
    """
    Parser per il file MASTER FAQ (data/master_faq/FAQ_ClassyFarm.md).

    Questo e' il sacro graal del RAG: risposte ufficiali curate con priorita' massima.

    Riconosce il formato:
        ## 1. Sezione Macro
        ### 1.2 Sottosezione
        #### Domanda: Testo domanda?
        **Risposta:** Testo risposta...
        ---

    Ogni coppia Q/A diventa un chunk con:
    - chunk_category: "master_faq"
    - source_type: "master_faq"
    - Contesto gerarchico preservato (sezione macro + sottosezione)
    """
    chunks = []

    if not MASTER_FAQ_DIR.exists():
        print(f"  Cartella MASTER FAQ non trovata: {MASTER_FAQ_DIR} (salto)")
        return chunks

    md_files = list(MASTER_FAQ_DIR.glob("*.md"))
    if not md_files:
        print(f"  Nessun file .md in {MASTER_FAQ_DIR}")
        return chunks

    for mfile in md_files:
        print(f"  Processing MASTER FAQ: {mfile.name}")

        try:
            content = mfile.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  ERRORE lettura {mfile.name}: {e}")
            continue

        # Parsing gerarchico
        current_macro = ""      # ## X. Titolo
        current_section = ""    # ### X.Y Titolo
        current_subsection = "" # #### Sottotitolo (se presente)

        lines = content.split("\n")
        i = 0
        file_chunks = 0

        while i < len(lines):
            line = lines[i].rstrip()

            # Sezione macro (##)
            if re.match(r'^##\s+\d+[\.\d]*\s*\.?\s+', line):
                current_macro = line.lstrip("#").strip()
                current_section = ""
            elif line.startswith("## ") and not line.startswith("### "):
                current_macro = line.lstrip("#").strip()
                current_section = ""

            # Sottosezione (###)
            elif line.startswith("### "):
                current_section = line.lstrip("#").strip()

            # Domanda (####)
            elif line.startswith("#### Domanda:") or line.startswith("#### Domanda"):
                question = line.replace("####", "").replace("Domanda:", "").replace("Domanda", "").strip()
                question = question.lstrip(":").strip()

                # Cerca la risposta nelle righe successive
                answer_lines = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].rstrip()

                    # Fine della risposta: nuovo separatore, nuova domanda, nuova sezione
                    if (next_line.startswith("---") or
                        next_line.startswith("#### Domanda") or
                        next_line.startswith("### ") or
                        next_line.startswith("## ")):
                        break

                    # Rimuovi marker "**Risposta:**" ma mantieni contenuto
                    cleaned = next_line.replace("**Risposta:**", "").strip()
                    if cleaned or next_line == "":
                        answer_lines.append(next_line)

                    j += 1

                # Costruisci la risposta
                answer = "\n".join(answer_lines).strip()
                # Rimuovi markers finali
                answer = re.sub(r'^\*\*Risposta:\*\*\s*', '', answer, flags=re.MULTILINE).strip()

                if question and answer:
                    # Testo del chunk con contesto gerarchico
                    text_parts = []
                    if current_macro:
                        text_parts.append(f"SEZIONE: {current_macro}")
                    if current_section:
                        text_parts.append(f"SOTTOSEZIONE: {current_section}")
                    text_parts.append(f"DOMANDA: {question}")
                    text_parts.append(f"RISPOSTA: {answer}")

                    text = "\n".join(text_parts)

                    # Determina ruolo semantico
                    q_lower = question.lower()
                    if any(kw in q_lower for kw in ["come", "dove", "quando", "quali passaggi", "procedura"]):
                        role = "procedural"
                    elif any(kw in q_lower for kw in ["cos'", "cosa", "significato", "definizione", "differenza"]):
                        role = "definitional"
                    else:
                        role = "informational"

                    # Estrai numero sezione (es. "1.2" da "1.2 Registrazione Operatore")
                    section_num_match = re.match(r'^(\d+(?:\.\d+)*)', current_section)
                    section_num = section_num_match.group(1) if section_num_match else ""

                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "parent_doc_id": hashlib.sha256(mfile.name.encode()).hexdigest()[:16],
                        "text_content": text,
                        "canonical_concepts": extract_concepts(text),
                        "metadata": {
                            "filename": mfile.name,
                            "page_number": None,
                            "source_type": "master_faq",
                            "semantic_role": role,
                            "chunk_category": "master_faq",
                            "word_count": len(text.split()),
                            "chunk_index": len(chunks),
                            "summary": question,
                            "keywords": extract_keywords(text),
                            "extracted_links": [],
                            "master_section_macro": current_macro,
                            "master_section": current_section,
                            "master_section_num": section_num,
                        }
                    })
                    file_chunks += 1

                i = j
                continue

            i += 1

        print(f"    {file_chunks} coppie Q/A estratte")

    print(f"  MASTER FAQ totale: {len(chunks)} chunk (priorita' massima)")
    return chunks


def process_faq_text_files() -> list:
    """
    Processa tutti i file .txt e .md nella cartella data/faq_input/.

    Formati supportati:

    # Formato 1 (Q: / A:)
    Q: Come accedo?
    A: Usa SPID o CIE.

    Q: Come mi registro?
    A: Compila il modulo...

    # Formato 2 (DOMANDA: / RISPOSTA:)
    DOMANDA: Come accedo?
    RISPOSTA: Usa SPID o CIE.

    # Formato 3 (Markdown ### + paragrafo)
    ### Come accedo?
    Usa SPID o CIE.

    ### Come mi registro?
    Compila il modulo...
    """
    chunks = []

    if not FAQ_INPUT_DIR.exists():
        print(f"  Cartella FAQ non trovata: {FAQ_INPUT_DIR} (salto)")
        return chunks

    files = list(FAQ_INPUT_DIR.rglob("*.txt")) + list(FAQ_INPUT_DIR.rglob("*.md"))
    if not files:
        print(f"  Nessun file FAQ in {FAQ_INPUT_DIR}")
        return chunks

    print(f"  Trovati {len(files)} file FAQ in {FAQ_INPUT_DIR}")

    # Regex per i 3 formati
    patterns = [
        # Q: ... A: ...
        re.compile(r'^\s*Q[:.]\s*(.+?)\n\s*A[:.]\s*(.+?)(?=\n\s*Q[:.]\s*|\Z)',
                   re.MULTILINE | re.DOTALL | re.IGNORECASE),
        # DOMANDA: ... RISPOSTA: ...
        re.compile(r'^\s*DOMANDA[:.]\s*(.+?)\n\s*RISPOSTA[:.]\s*(.+?)(?=\n\s*DOMANDA[:.]\s*|\Z)',
                   re.MULTILINE | re.DOTALL | re.IGNORECASE),
        # ### Domanda\n Risposta
        re.compile(r'^###+\s*(.+?)\n+(.+?)(?=\n###+\s*|\Z)',
                   re.MULTILINE | re.DOTALL),
    ]

    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  ERRORE lettura {fpath.name}: {e}")
            continue

        file_chunks = 0

        for pattern in patterns:
            matches = pattern.findall(content)
            if matches:
                for question, answer in matches:
                    question = question.strip()
                    answer = answer.strip()

                    if not question or not answer:
                        continue

                    text = f"DOMANDA: {question}\nRISPOSTA: {answer}"

                    role = "procedural" if any(kw in question.lower()
                        for kw in ["come", "dove", "quando", "quanto"]) else "definitional"

                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "parent_doc_id": hashlib.sha256(fpath.name.encode()).hexdigest()[:16],
                        "text_content": text,
                        "canonical_concepts": extract_concepts(text),
                        "metadata": {
                            "filename": fpath.name,
                            "page_number": None,
                            "source_type": "faq",
                            "semantic_role": role,
                            "chunk_category": "faq",
                            "word_count": len(text.split()),
                            "chunk_index": len(chunks),
                            "summary": question,
                            "keywords": extract_keywords(text),
                            "extracted_links": [],
                            "faq_section": fpath.stem.upper(),
                        }
                    })
                    file_chunks += 1

                break  # usa solo il primo formato che matcha

        print(f"  {fpath.name}: {file_chunks} FAQ")

    print(f"  TOTALE da file FAQ: {len(chunks)} chunk")
    return chunks


# ==================================
# MAIN
# ==================================

def main():
    print("=" * 70)
    print("  RICOSTRUZIONE KNOWLEDGE BASE PALES-AI")
    print("=" * 70)

    all_chunks = []

    # ----------------------------------------
    # STEP 0: Processa FAQ (priorita' massima)
    # ----------------------------------------
    print("\n[STEP 0] Processamento FAQ...")

    # 0a: MASTER FAQ (sacro graal, priorita' ASSOLUTA)
    master_faq_chunks = process_master_faq()
    all_chunks.extend(master_faq_chunks)

    # 0b: DOCX FAQ (formato Word)
    faq_path = BASE_DIR / "DOMANDE FREQUENTI.docx"
    faq_chunks = process_faq_docx(faq_path)
    all_chunks.extend(faq_chunks)

    # 0c: File FAQ TXT/MD aggiuntivi in data/faq_input/
    faq_text_chunks = process_faq_text_files()
    all_chunks.extend(faq_text_chunks)

    print(f"  FAQ totali: {len(master_faq_chunks)} master + {len(faq_chunks)} docx + {len(faq_text_chunks)} txt/md = {len(master_faq_chunks) + len(faq_chunks) + len(faq_text_chunks)} chunk")

    # ----------------------------------------
    # STEP 1: Processa PDF
    # ----------------------------------------
    print("\n[STEP 1] Processamento PDF...")
    pdfs = collect_all_pdfs()
    print(f"  Trovati {len(pdfs)} PDF in {PDF_INPUT_DIR}")

    pdf_chunks = 0
    for i, pdf_path in enumerate(pdfs, 1):
        rel_path = pdf_path.relative_to(PDF_INPUT_DIR)
        print(f"  [{i}/{len(pdfs)}] {rel_path}")

        pages = extract_pdf_pages(pdf_path)
        if not pages:
            print(f"    Nessun testo estratto")
            continue

        for page in pages:
            chunks = smart_chunk_text(
                text=page["text"],
                filename=page["filename"],
                page_number=page["page_number"],
                source_type="pdf"
            )
            all_chunks.extend(chunks)
            pdf_chunks += len(chunks)

    print(f"\n  PDF: {pdf_chunks} chunk generati da {len(pdfs)} file")

    # ----------------------------------------
    # STEP 2: Recupera chunk web utili
    # ----------------------------------------
    print("\n[STEP 2] Recupero chunk web esistenti...")
    web_chunks = load_existing_web_chunks()

    # Filtra spazzatura anche dai web chunk
    clean_web = []
    for chunk in web_chunks:
        text = chunk.get("text_content", "")
        if not is_junk(text) and len(text.split()) >= MIN_CHUNK_WORDS:
            # Aggiungi word_count se mancante
            if not chunk.get("metadata", {}).get("word_count"):
                chunk.setdefault("metadata", {})["word_count"] = len(text.split())
            chunk["metadata"]["source_type"] = "web"
            clean_web.append(chunk)

    print(f"  Web: {len(clean_web)} chunk utili (da {len(web_chunks)} originali)")
    all_chunks.extend(clean_web)

    # ----------------------------------------
    # STEP 3: Deduplicazione
    # ----------------------------------------
    print(f"\n[STEP 3] Deduplicazione...")
    print(f"  Chunk totali prima: {len(all_chunks)}")
    all_chunks = deduplicate_chunks(all_chunks)
    print(f"  Chunk totali dopo: {len(all_chunks)}")

    # ----------------------------------------
    # STEP 4: Statistiche qualita'
    # ----------------------------------------
    print(f"\n[STEP 4] Statistiche qualita':")

    word_counts = [len(c["text_content"].split()) for c in all_chunks]
    roles = {}
    categories = {}
    sources = {"pdf": 0, "web": 0}

    for c in all_chunks:
        role = c.get("metadata", {}).get("semantic_role", "unknown")
        cat = c.get("metadata", {}).get("chunk_category", "unknown")
        src = c.get("metadata", {}).get("source_type", "unknown")
        roles[role] = roles.get(role, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
        sources[src] = sources.get(src, 0) + 1

    print(f"  Chunk totali: {len(all_chunks)}")
    print(f"  Parole - min: {min(word_counts)}, max: {max(word_counts)}, "
          f"media: {sum(word_counts) // len(word_counts)}")
    print(f"  Per sorgente: {sources}")
    print(f"  Per ruolo: {roles}")
    print(f"  Per categoria: {categories}")

    # ----------------------------------------
    # STEP 5: Salvataggio dataset
    # ----------------------------------------
    print(f"\n[STEP 5] Salvataggio dataset...")
    chunks_file = save_dataset(all_chunks)

    # ----------------------------------------
    # STEP 6: Ricostruzione indici
    # ----------------------------------------
    print(f"\n[STEP 6] Ricostruzione indici...")
    rebuild = input("\n  Vuoi ricostruire gli indici FAISS + BM25 + SQLite? (s/n): ").strip().lower()

    if rebuild == "s":
        rebuild_indices(chunks_file)
    else:
        print("  Indicizzazione saltata. Esegui manualmente quando pronto.")

    # ----------------------------------------
    # DONE
    # ----------------------------------------
    print("\n" + "=" * 70)
    print("  RICOSTRUZIONE COMPLETATA")
    print(f"  Dataset: {chunks_file}")
    print(f"  Chunk totali: {len(all_chunks)}")
    print(f"  PDF: {sources.get('pdf', 0)} | Web: {sources.get('web', 0)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
