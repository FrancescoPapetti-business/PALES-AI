"""
layer5_validation_governance.py
Quality Gate Finale: Valida schema, contenuto, metadati e sicurezza (PII).
"""

import sys
import re
from typing import Tuple, List, Dict, Any
from pathlib import Path

# Gestione path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# --- CONFIGURAZIONE GOVERNANCE ---

# Frasi di rifiuto LLM (Allucinazioni di rifiuto)
REFUSAL_PHRASES = [
    "sembra che il testo", "as an ai", "come modello di linguaggio",
    "i cannot rewrite", "non posso riscrivere", "non posso completare",
    "sorry, i cannot", "mi dispiace, ma non posso", "non sono in grado di"
]

# Regex PII (Dati Sensibili Italiani - Semplificati per Governance)
# Utile per evitare di indicizzare dati personali reali per errore
REGEX_EMAIL = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
REGEX_CF = r"[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]" # Codice Fiscale pattern base

def check_content_quality(text: str) -> List[str]:
    """Verifica la qualità testuale del singolo chunk."""
    issues = []
    
    if not text:
        return ["Testo vuoto."]
    
    # 1. Lunghezza Minima (Abbassata a 50 per descrizioni visive brevi ma valide)
    if len(text) < 50:
        issues.append(f"Testo troppo breve ({len(text)} chars).")

    # 2. Artefatti LLM
    text_lower = text.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in text_lower:
            issues.append(f"Rifiuto LLM rilevato: '{phrase}'")
            break

    return issues

def check_governance_security(text: str) -> List[str]:
    """Verifica presenza di PII (Email, CF)."""
    issues = []
    
    # Check Email
    # Check Email DISABILITATO: email istituzionali ClassyFarm non sono PII
    # if re.search(REGEX_EMAIL, text):
    #     issues.append("⚠️ WARN: Possibile Email rilevata nel testo.")      
    # Check Codice Fiscale (Solo warning, a volte sono codici tecnici simili)
    if re.search(REGEX_CF, text):
        issues.append("⚠️ WARN: Possibile Codice Fiscale rilevato.")
        
    return issues

def check_schema_integrity(chunk: Dict) -> List[str]:
    """Verifica che il chunk abbia gli ID e i metadati necessari per il Vector Store."""
    issues = []
    
    required_keys = ["chunk_id", "parent_doc_id", "text_content", "metadata"]
    for k in required_keys:
        if k not in chunk or not chunk[k]:
            issues.append(f"Chiave mancante o nulla: {k}")
            
    # Verifica Metadati specifici (generati dal Layer 4)
    meta = chunk.get("metadata", {})
    if not isinstance(meta, dict):
        issues.append("Formato metadata non valido (deve essere dict).")
    else:
        if "source_type" not in meta:
            issues.append("Metadata mancante: source_type")
            
    return issues

def classify_chunk_category(text: str, meta: Dict) -> str:
    """
    Assegna una categoria al chunk per aiutare il filtering nel retrieval.
    """
    text_lower = text.lower()

    # 0. Pagine web ClassyFarm — categoria dedicata
    if meta.get("source_type") == "web_page":
        section = (meta.get("web_section") or "").lower()
        if section == "faq":
            return "web_faq"
        if section in ("guide", "manuali"):
            return "web_guide"
        if section in ("normativa", "documentazione"):
            return "web_normativa"
        return "web_content"
    
    # 1. Tabelle Sanzioni / Checklist (Pattern visivi o keyword)
    if "sanzione" in text_lower and ("euro" in text_lower or "€" in text_lower):
        return "sanctions_table"
    if text_lower.count("[ ]") > 3 or text_lower.count("☐") > 3:
        return "checklist"
    
    # 2. Definizioni / Glossario
    if meta.get("source_type") == "faq" or "significa" in text_lower or "definizione" in text_lower:
        return "glossary"
        
    # 3. Procedurale
    if "cliccare" in text_lower or "selezionare" in text_lower or "accedere" in text_lower:
        return "procedural"
        
    return "other"

def validate_chunk_batch(chunks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Processa una lista di chunk.
    Returns:
        valid_chunks: Lista di chunk pronti per il DB.
        rejected_chunks: Lista di chunk scartati con motivo (per logging).
    """
    valid_chunks = []
    rejected_chunks = []

    if not isinstance(chunks, list):
        print("❌ Errore Critico: Layer 5 si aspettava List[Dict], ricevuto altro.")
        return [], []

    print(f"🛡️  Governance Check su {len(chunks)} chunks...")

    for chunk in chunks:
        chunk_issues = []
        
        # 1. Validazione Schema (Tecnica)
        chunk_issues.extend(check_schema_integrity(chunk))
        
        # Se lo schema è rotto, non procediamo oltre
        if not chunk_issues:
            text = chunk.get("text_content", "")
            
            # 2. Validazione Contenuto (Qualità)
            chunk_issues.extend(check_content_quality(text))
            
            # 3. Validazione Sicurezza (Governance)
            # Nota: I warning PII non scartano il chunk ma lo loggano (o aggiungono flag)
            security_warnings = check_governance_security(text)
            if security_warnings:
                # Aggiungiamo un flag ai metadata per filtraggio futuro lato app
                chunk["metadata"]["pii_warning"] = True
                print(f"   [SEC] Chunk {chunk.get('chunk_id')[:8]}: {security_warnings}")
            
            # 4. Categorizzazione (Nuovo per Layer 7 filtering)
            category = classify_chunk_category(text, chunk.get("metadata", {}))
            chunk["metadata"]["chunk_category"] = category

        if chunk_issues:
            # Chunk Scartato
            rejected_record = {
                "chunk_id": chunk.get("chunk_id", "UNKNOWN"),
                "issues": chunk_issues,
                "preview": chunk.get("text_content", "")[:50]
            }
            rejected_chunks.append(rejected_record)
        else:
            # Chunk Approvato
            chunk["metadata"]["validated_at"] = True # Timbro di validazione
            valid_chunks.append(chunk)

    return valid_chunks, rejected_chunks

def is_answerable_from_docs(retrieved_contexts: List[str], question: str) -> str:
    """
    Helper euristico per evaluation (RAGAS).
    Ritorna: 'likely_answerable', 'retrieval_failure', 'knowledge_gap'
    """
    if not retrieved_contexts:
        return "retrieval_failure"
        
    q_words = set(re.findall(r"\w+", question.lower()))
    # Filtra stop words banali (molto basic)
    q_words = {w for w in q_words if len(w) > 3}
    
    combined_ctx = " ".join(retrieved_contexts).lower()
    
    # Se nessuna parola chiave della domanda è nel contesto -> Retrieval Failure
    if not any(w in combined_ctx for w in q_words):
        return "retrieval_failure"
        
    # Altrimenti assumiamo che sia answerable o knowledge gap (difficile distinguere senza LLM)
    return "likely_answerable"

if __name__ == "__main__":
    # Test Standalone con Mock Data dal Layer 4
    print("🔍 Testing Layer 5 (Governance)...")
    
    mock_chunks_layer4 = [
        # Caso 1: Valido
        {
            "chunk_id": "uuid-1234",
            "parent_doc_id": "doc-001",
            "text_content": "Per accedere al sistema ClassyFarm inserire username e password nel box a destra.",
            "metadata": {"source_type": "text_document", "keywords": ["login"]}
        },
        # Caso 2: Rifiuto LLM (Deve fallire)
        {
            "chunk_id": "uuid-5678",
            "parent_doc_id": "doc-001",
            "text_content": "Mi dispiace, ma come modello di linguaggio non posso descrivere l'immagine.",
            "metadata": {"source_type": "visual_guide"}
        },
        # Caso 3: Schema Rotto (Deve fallire)
        {
            "text_content": "Manca l'ID qui."
        },
        # Caso 4: PII Detect (Deve passare ma con Warning)
        {
            "chunk_id": "uuid-9999",
            "parent_doc_id": "doc-002",
            "text_content": "Contattare il responsabile all'indirizzo mario.rossi@ministero.it per info.",
            "metadata": {"source_type": "contact_info"}
        }
    ]

    valid, rejected = validate_chunk_batch(mock_chunks_layer4)

    print(f"\n✅ Chunks Validi: {len(valid)}")
    print(f"❌ Chunks Scartati: {len(rejected)}")

    if rejected:
        print("\nDettaglio Scarti:")
        for r in rejected:
            print(f" - ID: {r['chunk_id']} | Motivo: {r['issues']}")
            
    if valid:
        pii_flagged = [c for c in valid if c["metadata"].get("pii_warning")]
        if pii_flagged:
            print(f"\n⚠️ Chunks validi con Warning PII: {len(pii_flagged)}")