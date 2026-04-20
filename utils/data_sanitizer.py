"""
utils/data_sanitizer.py
Modulo centralizzato per la gestione della Privacy (Mascheramento PII) 
e l'Audit Trail (Log Hashing).
"""
import json
import re
import hashlib
import os
from pathlib import Path
from typing import Dict

# ===========================================================
# COSTANTI E PATTERN (PII)
# ===========================================================
PII_PATTERNS = {
    # Codice Fiscale Italiano (Pattern robusto)
    "codice_fiscale": r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
    # Telefono (vari formati, italiano e generico)
    "phone": r"\b\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}\b",
    # Carte di credito (semplificato)
    "credit_card": r"\b(?:\d[ -]*?){13,16}\b", 
}

# ===========================================================
# LOGICA PII MASKING
# ===========================================================

def redact_pii(text: str) -> str:
    """Oscura i dati personali trovati nel testo."""
    if not text:
        return ""
        
    safe_text = text
    
    safe_text = re.sub(PII_PATTERNS["phone"], "[PHONE_MASKED]", safe_text)
    safe_text = re.sub(PII_PATTERNS["codice_fiscale"], "[CF_MASKED]", safe_text)
    safe_text = re.sub(PII_PATTERNS["credit_card"], "[CARD_MASKED]", safe_text)
    
    return safe_text

# ===========================================================
# LOGICA AUDIT (LOG HASHING)
# ===========================================================

def calculate_sha256_hash(data: str) -> str:
    """Calcola l'hash SHA256 di una stringa."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def get_last_log_hash(log_path: Path) -> str:
    """
    Recupera l'hash del log precedente per l'audit chain.
    In un ambiente di produzione, questo dovrebbe usare un database dedicato, 
    ma qui leggiamo l'ultima riga del file JSONL.
    """
    if not log_path.exists():
        return "0" * 64 # Hash iniziale (64 zeri)

    try:
        # Legge l'ultima riga del file (la più veloce)
        with open(log_path, 'rb') as f:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode('utf-8').strip()
        
        # Estrae l'hash dal JSON dell'ultima riga
        if last_line:
            last_entry = json.loads(last_line)
            return last_entry.get("audit_hash", "0" * 64)
        
    except Exception:
        # File vuoto, corrotto o troppo piccolo. Ricomincia la catena.
        pass
        
    return "0" * 64

def finalize_audit_entry(log_entry: Dict, log_path: Path) -> Dict:
    """
    Aggiunge il previous_hash e calcola l'hash finale (per l'Audit Trail).
    """
    # 1. Recupera l'hash della riga precedente
    previous_hash = get_last_log_hash(log_path)
    
    # 2. Aggiungi il campo alla entry (per il log strutturato)
    log_entry["previous_audit_hash"] = previous_hash
    
    # 3. Serializza l'entry per calcolare l'hash (deve essere ordinata per essere riproducibile)
    entry_data = json.dumps(log_entry, sort_keys=True)
    
    # 4. Calcola l'hash di questa nuova riga
    current_hash = calculate_sha256_hash(entry_data)
    
    # 5. Aggiungi l'hash finale alla entry
    log_entry["audit_hash"] = current_hash
    
    return log_entry