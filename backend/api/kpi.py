"""
backend/api/kpi.py
Gestione Log KPI.
Delega la configurazione del logging a utils.logging_config per coerenza.
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# ==========================
#   SETUP IMPORT
# ==========================
# backend/api/kpi.py -> parents[2] = root (classyfarm-rag)
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Importa il logger e il percorso file centralizzati
# Se questo import fallisce, assicurati di aver aggiornato utils/logging_config.py
from utils.logging_config import get_kpi_logger, KPI_LOG_PATH
from utils.data_sanitizer import redact_pii, finalize_audit_entry

# Ottieni l'istanza del logger già configurata (formato JSONL raw)
kpi_logger = get_kpi_logger()


def track_interaction(
    endpoint: str,
    status: str,
    latency: float,
    user_role: str,
    metadata: dict = None
):
    """
    Scrive una riga di log strutturato.
    Non serve formattare qui o aggiungere handler, ci pensa kpi_logger.
    """

    # 1. Applicazione della Logica PII:
    if metadata and "user_query" in metadata:
        metadata["user_query_masked"] = redact_pii(metadata["user_query"])
        # Rimuoviamo la query originale per sicurezza, se l'audit non la richiede
        del metadata["user_query"]

    event = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "status": status,
        "latency_sec": latency,
        "user_role": user_role,
        "meta": metadata or {}
    }
    
    # 2. Applicazione della Logica Audit Hashing:
    # Trasforma l'entry, calcola gli hash e aggiunge previous_audit_hash e audit_hash
    final_entry = finalize_audit_entry(event, KPI_LOG_PATH)

    # 3.Scrittura diretta del JSON string.
    # Il logger è configurato con un formatter '%(message)s', quindi scriverà solo questo JSON.
    kpi_logger.info(json.dumps(event))


def read_latest_kpi(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Legge le ultime righe del file di log usando il percorso centralizzato.
    """
    metrics = []
    
    # Usiamo KPI_LOG_PATH importato da utils, così la "fonte di verità" è unica
    if not KPI_LOG_PATH.exists():
        return []

    try:
        with open(KPI_LOG_PATH, "r", encoding="utf-8") as f:
            # Legge tutte le righe e prende le ultime 'limit'
            lines = f.readlines()[-limit:]
            for line in lines:
                try:
                    metrics.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        # Inverte l'ordine (dal più recente al più vecchio) per la visualizzazione
        return metrics[::-1]
        
    except Exception as e:
        print(f"Errore lettura KPI: {e}")
        return []