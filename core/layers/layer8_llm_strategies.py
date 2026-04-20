"""
LLM Generation Strategies - Layer 8

Strategie diverse per intent:
- Direct LLM: definitional/acronym (massima comprensione semantica)
- Procedural Synthesis: procedural (step-by-step guidato)
"""

import logging
from typing import List, Dict, Tuple
from langchain_core.documents import Document

logger = logging.getLogger("layer8_strategies")

# ============================================
# STRATEGY 1: DIRECT LLM (Simple RAG-style)
# ============================================

def llm_direct_answer(
    question: str,
    docs: List[Document],
    llm,
    chat_history: List[Dict] = None
) -> Tuple[str, List[Document]]:
    """
    Genera risposta usando LLM direttamente sul contesto completo.
    
    Filosofia: Lascia che LLM comprenda semanticamente senza filtri.
    Come Simple RAG ma con BM25 boost.
    
    Usato per: definitional, acronym_definition
    """
    logger.info("🤖 LLM Direct Strategy (definitional/acronym)")
    
    # Import lazy per evitare circular dependencies
    from core.layers.layer8_application import (
        format_docs_for_prompt,
        format_links_for_prompt,
        format_chat_history,
        get_system_prompt,
        FULL_PROMPT_TEMPLATE,
        enrich_context_with_temporal_note
    )
    
    # Prepara contesto COMPLETO (NO evidence filtering)
    context_str_raw = format_docs_for_prompt(docs)
    context_str = enrich_context_with_temporal_note(context_str_raw)
    links_str = format_links_for_prompt(docs)
    system_prompt = get_system_prompt()
    
    # Prompt standard (LLM decide cosa è rilevante)
    final_prompt = FULL_PROMPT_TEMPLATE.format(
        system_prompt_text=system_prompt,
        context_str=context_str,
        links_str=links_str,
        chat_history_str=format_chat_history(chat_history or []),
        question=question
    )
    
    # Invoca LLM
    response = llm.invoke(final_prompt)
    answer = response.content.strip()
    
    logger.info(f"✅ LLM Direct answer generated ({len(answer)} chars)")
    
    return answer, docs


# ============================================
# STRATEGY 2: PROCEDURAL SYNTHESIS
# ============================================

def llm_procedural_synthesis(
    question: str,
    docs: List[Document],
    llm
) -> Tuple[str, List[Document]]:
    """
    Sintesi procedurale step-by-step per domande how-to.
    
    Usa prompt specializzato per estrarre procedure.
    
    Usato per: procedural
    """
    logger.info("🔧 Procedural Synthesis Strategy")
    
    from core.layers.layer8_application import synthesize_procedure_from_docs
    
    # Genera prompt procedurale
    proc_prompt = synthesize_procedure_from_docs(docs, question)
    
    # Invoca LLM con prompt specializzato
    response = llm.invoke(proc_prompt)
    answer = response.content.strip()
    
    logger.info(f"✅ Procedural answer generated ({len(answer)} chars)")
    
    return answer, docs