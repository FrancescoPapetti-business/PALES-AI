"""
Layer 2.5: Query Expansion
Expand user queries with synonyms, reformulations, and context enrichment.
"""

import logging
from typing import List, Dict, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("layer2_query_expansion")

# ============================================
# SYNONYM DICTIONARY (Domain-Specific)
# ============================================

DOMAIN_SYNONYMS = {
    # Acronimi
    "odc": ["organismo di certificazione", "ente certificatore"],
    "bdn": ["banca dati nazionale", "database nazionale"],
    "sqnba": ["sistema qualità nazionale benessere animale"],
    "pac": ["politica agricola comune"],
    
    # Termini tecnici
    "operatore": ["allevatore", "detentore", "proprietario"],
    "delegato": ["rappresentante", "incaricato"],
    "veterinario aziendale": ["veterinario di azienda", "vet aziendale"],
    "veterinario incaricato": ["veterinario delegato", "vet incaricato"],
    
    # Azioni procedurali
    "associare": ["collegare", "abbinare", "registrare"],
    "compilare": ["riempire", "completare"],
    "caricare": ["uploadare", "inviare", "trasmettere"],
    "scaricare": ["downloadare", "salvare", "esportare"],
    "modificare": ["cambiare", "aggiornare", "editare"],
    "cancellare": ["eliminare", "rimuovere", "delete"],
    
    # Documenti/Entità
    "checklist": ["cl", "check-list", "questionario"],
    "cruscotto": ["dashboard", "pannello di controllo"],
    "allevamento": ["azienda", "struttura", "impresa"],
    "abilitazione": ["autorizzazione", "permesso", "accesso"],
}

# ============================================
# QUERY EXPANSION CLASS
# ============================================

class QueryExpander:
    """
    Expand queries using multiple strategies:
    1. Synonym replacement (dictionary-based)
    2. LLM-based reformulation (semantic)
    3. Context enrichment (intent-aware)
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Args:
            llm: Optional LLM for semantic reformulation
        """
        self.llm = llm
        self.synonyms = DOMAIN_SYNONYMS
        
        # LLM prompt for reformulation
        self.reformulation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Sei un assistente esperto del sistema ClassyFarm per il benessere animale.
Ricevi una domanda dell'utente e devi riformularla in 2 modi diversi mantenendo lo stesso significato:
1. Una versione più tecnica/formale
2. Una versione più colloquiale/semplificata

Rispondi SOLO con le 2 riformulazioni separate da "|||", senza numeri o spiegazioni.

Esempio:
Input: "Come faccio a scaricare il report?"
Output: "Qual è la procedura per l'esportazione del report?|||Dove trovo il pulsante per scaricare il report?"
"""),
            ("user", "{query}")
        ])
    
    # ========================================
    # STRATEGY 1: SYNONYM EXPANSION
    # ========================================
    
    def expand_with_synonyms(self, query: str, max_expansions: int = 2) -> List[str]:
        """
        Replace keywords with synonyms from dictionary.
        
        Args:
            query: Original query
            max_expansions: Max number of synonym variants to generate
        
        Returns:
            List of expanded queries (including original)
        """
        expanded = [query]  # Always include original
        
        query_lower = query.lower()
        
        # Find matches
        for term, synonyms in self.synonyms.items():
            if term in query_lower:
                # Generate variants by replacing term with each synonym
                for syn in synonyms[:max_expansions]:
                    variant = query_lower.replace(term, syn)
                    expanded.append(variant.capitalize())
                break  # Only expand first match to avoid combinatorial explosion
        
        return list(set(expanded))[:max_expansions + 1]  # Deduplicate + limit
    
    # ========================================
    # STRATEGY 2: LLM REFORMULATION
    # ========================================
    
    def expand_with_llm(self, query: str) -> List[str]:
        """
        Use LLM to generate semantic reformulations.
        
        Args:
            query: Original query
        
        Returns:
            List of reformulated queries
        """
        if not self.llm:
            logger.warning("LLM not available for reformulation")
            return []
        
        try:
            prompt = self.reformulation_prompt.format_messages(query=query)
            response = self.llm.invoke(prompt)
            
            # Parse response
            reformulations = response.content.strip().split("|||")
            reformulations = [r.strip() for r in reformulations if r.strip()]
            
            logger.info(f"✅ LLM reformulations: {len(reformulations)}")
            return reformulations
        
        except Exception as e:
            logger.error(f"❌ LLM reformulation failed: {e}")
            return []
    
    # ========================================
    # STRATEGY 3: INTENT-BASED ENRICHMENT
    # ========================================
    
    def expand_with_intent(self, query: str, intent: str) -> List[str]:
        """
        Add context based on detected intent.
        
        Args:
            query: Original query
            intent: Detected intent (from layer2_intent)
        
        Returns:
            List of enriched queries
        """
        enriched = []
        
        # Intent-specific enrichment templates
        enrichment_map = {
            "procedural": [
                f"procedura per {query}",
                f"come si fa a {query}",
                f"passaggi per {query}"
            ],
            "definitional": [
                f"definizione di {query}",
                f"cosa significa {query}",
                f"spiegazione {query}"
            ],
            "acronym_definition": [
                f"cosa significa {query}",
                f"significato acronimo {query}",
                f"per cosa sta {query}"
            ]
        }
        
        if intent in enrichment_map:
            templates = enrichment_map[intent]
            for template in templates[:2]:  # Limit to 2 variants
                enriched.append(template)
        
        return enriched
    
    # ========================================
    # MAIN EXPANSION METHOD
    # ========================================
    
    def expand(
        self,
        query: str,
        intent: Optional[str] = None,
        strategies: List[str] = ["synonyms", "llm", "intent"]
    ) -> Dict[str, List[str]]:
        """
        Expand query using multiple strategies.
        
        Args:
            query: Original query
            intent: Detected intent (optional)
            strategies: List of strategies to use
        
        Returns:
            Dict mapping strategy name to list of expanded queries
        """
        results = {
            "original": [query],
            "synonyms": [],
            "llm": [],
            "intent": [],
            "all": [query]  # Combined list (deduplicated)
        }
        
        # Strategy 1: Synonyms
        if "synonyms" in strategies:
            logger.info("🔍 Expanding with synonyms...")
            syn_queries = self.expand_with_synonyms(query)
            results["synonyms"] = [q for q in syn_queries if q != query]
            results["all"].extend(results["synonyms"])
        
        # Strategy 2: LLM reformulation
        if "llm" in strategies and self.llm:
            logger.info("🤖 Expanding with LLM reformulation...")
            llm_queries = self.expand_with_llm(query)
            results["llm"] = llm_queries
            results["all"].extend(llm_queries)
        
        # Strategy 3: Intent enrichment
        if "intent" in strategies and intent:
            logger.info(f"🎯 Expanding with intent={intent}...")
            intent_queries = self.expand_with_intent(query, intent)
            results["intent"] = intent_queries
            results["all"].extend(intent_queries)
        
        # Deduplicate combined list
        results["all"] = list(set(results["all"]))
        
        logger.info(f"📊 Expansion summary:")
        logger.info(f"  - Original: 1")
        logger.info(f"  - Synonyms: {len(results['synonyms'])}")
        logger.info(f"  - LLM: {len(results['llm'])}")
        logger.info(f"  - Intent: {len(results['intent'])}")
        logger.info(f"  - Total unique: {len(results['all'])}")
        
        return results

# ============================================
# UTILITY FUNCTION
# ============================================

def expand_query(
    query: str,
    llm: Optional[BaseChatModel] = None,
    intent: Optional[str] = None,
    max_queries: int = 5
) -> List[str]:
    """
    Convenience function for query expansion.
    
    Args:
        query: Original query
        llm: Optional LLM for reformulation
        intent: Detected intent
        max_queries: Maximum number of queries to return
    
    Returns:
        List of expanded queries (including original)
    """
    expander = QueryExpander(llm=llm)
    results = expander.expand(query, intent=intent)
    
    # Return top-N unique queries
    return results["all"][:max_queries]

# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test cases
    test_queries = [
        ("Come faccio ad associare il mio veterinario?", "procedural"),
        ("Cosa si intende per OdC?", "acronym_definition"),
        ("Qual è la definizione di operatore?", "definitional"),
        ("Come scaricare il report SQNBA?", "procedural"),
    ]
    
    print("\n🧪 Testing Query Expansion\n" + "="*60)
    
    for query, intent in test_queries:
        print(f"\n📝 Original: {query}")
        print(f"🎯 Intent: {intent}")
        
        # Without LLM
        expander = QueryExpander(llm=None)
        results = expander.expand(query, intent=intent, strategies=["synonyms", "intent"])
        
        print(f"\n📊 Expansions:")
        for i, exp_q in enumerate(results["all"], 1):
            print(f"  {i}. {exp_q}")
        
        print("-" * 60)