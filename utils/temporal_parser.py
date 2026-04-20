"""
utils/temporal_parser.py
Estrazione e normalizzazione di entità temporali per preservare precisione.
"""
import re
from typing import List, Dict, Optional

# Pattern per entità temporali (italiano)
TEMPORAL_PATTERNS = {
    "giorni_numero": r"\b(\d+)\s*(giorni?|gg\.?)\b",
    "ore_numero": r"\b(\d+)\s*(ore?|h)\b",
    "settimane_numero": r"\b(\d+)\s*(settimane?)\b",
    "mesi_numero": r"\b(\d+)\s*(mesi?)\b",
    "entro_generico": r"\bentro\s+(\d+)\s*(giorni?|ore?|settimane?|mesi?)\b",
    "data_specifica": r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
    "orario": r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b",
    "giorni_lavorativi": r"\b(\d+)\s*giorni?\s+lavorativ[ie]\b",
}

def extract_temporal_entities(text: str) -> List[Dict[str, any]]:
    """
    Estrae tutte le entità temporali dal testo.
    
    Returns:
        List[Dict] con chiavi: {type, value, matched_text, position}
    """
    entities = []
    text_lower = text.lower()
    
    for entity_type, pattern in TEMPORAL_PATTERNS.items():
        for match in re.finditer(pattern, text_lower, re.IGNORECASE):
            entities.append({
                "type": entity_type,
                "value": match.group(0),
                "matched_text": text[match.start():match.end()],  # Mantieni case originale
                "position": match.span(),
                "numeric_value": match.group(1) if match.groups() else None
            })
    
    # Ordina per posizione nel testo
    entities.sort(key=lambda x: x["position"][0])
    
    # Deduplica overlap (es. "10 giorni lavorativi" catturato da 2 pattern)
    deduplicated = []
    used_ranges = set()
    for entity in entities:
        span = entity["position"]
        if span not in used_ranges:
            deduplicated.append(entity)
            used_ranges.add(span)
    
    return deduplicated

def highlight_temporal_in_text(text: str) -> str:
    """
    Aggiunge marker ⏰ prima delle entità temporali nel testo.
    Usato per enfatizzare al LLM l'importanza di preservarle.
    """
    entities = extract_temporal_entities(text)
    
    if not entities:
        return text
    
    # Applica marker in ordine inverso per non sballare posizioni
    marked_text = text
    for entity in reversed(entities):
        start, end = entity["position"]
        marked_text = (
            marked_text[:start] + 
            f"⏰{entity['matched_text']}" + 
            marked_text[end:]
        )
    
    return marked_text

def validate_temporal_preservation(answer: str, context: str) -> tuple[bool, List[str]]:
    """
    Verifica che le entità temporali del context siano presenti nell'answer.
    
    Returns:
        (is_valid, missing_entities)
    """
    context_temporal = extract_temporal_entities(context)
    answer_temporal = extract_temporal_entities(answer)
    
    # Set di valori numerici estratti
    context_values = {e["value"].lower() for e in context_temporal}
    answer_values = {e["value"].lower() for e in answer_temporal}
    
    missing = context_values - answer_values
    
    return (len(missing) == 0, list(missing))

def enrich_context_with_temporal_note(context: str) -> str:
    """
    Aggiunge nota esplicativa se il contesto contiene valori temporali.
    """
    entities = extract_temporal_entities(context)
    
    if not entities:
        return context
    
    temporal_values = [e["matched_text"] for e in entities[:5]]  # Max 5
    note = f"\n\n⏰ ATTENZIONE: Questo contesto contiene valori temporali specifici: {', '.join(temporal_values)}. Devono essere riportati ESATTAMENTE nella risposta.\n"
    
    return note + context