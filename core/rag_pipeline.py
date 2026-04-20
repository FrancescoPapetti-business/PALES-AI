import sys
from pathlib import Path
from typing import Optional, Dict, Any

# ==========================
#   SETUP PATH
# ==========================
# Risolve la root 'classyfarm-rag' per importare core.layers
# core/rag_pipeline.py -> parents[1] -> classyfarm-rag (root)
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Importazione corretta dalla nuova struttura
from core.layers.layer8_application import process_request
from core.layers.layer4_semantic_rewriting import expand_query_semantic

def is_failed_answer(result: Dict[str, Any]) -> bool:
    if not result:
        return True

    answer = result.get("answer", "").lower()
    sources = result.get("sources", [])

    if not sources:
        return True

    if "non è presente" in answer:
        return True

    # Fix G: Intercettare rifiuti reali
    lower_ans = answer.lower()
    refusal_phrases = ["non emerge", "non ho trovato informazioni", "non è specificato", "mi dispiace"]
    if any(p in lower_ans for p in refusal_phrases):
        return True

    return False



def rag_pipeline(query: str, role: Optional[str] = None) -> Dict[str, Any]:
    effective_role = role if role else "operatore"

    # === Tentativo 1 ===
    result = process_request(
        question=query,
        role=effective_role
    )

    if not is_failed_answer(result):
        result["orchestration_path"] = ["initial"]
        return result

    # === Tentativo 2: retry IDENTICO (serve per cross-doc / ranking) ===
    expanded_query = expand_query_semantic(query)
    result_retry = process_request(
        question=expanded_query, # Fix F: Usa expanded_query
        role=effective_role
    )

    if not is_failed_answer(result_retry):
        result_retry["orchestration_path"] = ["initial", "retry_same_query"]
        return result_retry

    # === Fallimento reale ===
    result["orchestration_path"] = ["initial", "retry_same_query", "not_found"]
    return result
