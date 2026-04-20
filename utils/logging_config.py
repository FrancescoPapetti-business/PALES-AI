"""
utils/logging_config.py
Modulo centralizzato di logging per PALES-AI.
Gestisce:
1. Log Applicativi (Debug, Errori) -> logs/chatbot.log (Testo leggibile)
2. Log KPI (Metriche, Audit) -> logs/kpi_metrics.jsonl (JSON strutturato)
"""

import sys
import logging
from pathlib import Path

# ==========================
#   SETUP PERCORSI
# ==========================
# Risolve la root del progetto: utils -> parents[1] -> root
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Percorsi File
APP_LOG_PATH = LOG_DIR / "chatbot.log"
KPI_LOG_PATH = LOG_DIR / "kpi_metrics.jsonl"


def get_logger(name: str) -> logging.Logger:
    """
    Logger applicativo standard per debug ed errori.
    Formato: Timestamp | Nome | Livello | Messaggio
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evita duplicazione handler se richiamato più volte
    if not logger.handlers:
        # Handler su File
        file_handler = logging.FileHandler(APP_LOG_PATH, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Opzionale: Handler su Console (per vedere i log mentre sviluppi)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_kpi_logger() -> logging.Logger:
    """
    Logger specifico per i KPI.
    CRUCIALE: Usa un formatter 'raw' (solo %(message)s).
    Non aggiunge timestamp o livelli perché il messaggio in input è già
    un JSON completo generato dal backend.
    """
    logger = logging.getLogger("classyfarm_kpi")
    logger.setLevel(logging.INFO)
    
    # Evita che i log KPI finiscano anche nella console o nel root logger
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(KPI_LOG_PATH, encoding="utf-8")
        
        # ⚠️ FORMATTER CRUCIALE: Solo il messaggio.
        # Se aggiungiamo date o livelli qui, rompiamo il JSONL.
        formatter = logging.Formatter("%(message)s")
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

