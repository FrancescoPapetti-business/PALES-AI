"""
Test Empirico: RAG Semplice vs Sistema Attuale
Confronto diretto su domande critiche per valutare se la complessità è giustificata.
"""

import sys
from pathlib import Path
from typing import Dict, List
import time

# Setup path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Import sistema attuale per confronto
from core.layers.layer8_application import process_request

# ============================================
# RAG SEMPLICE (50 righe)
# ============================================

class SimpleRAG:
    """RAG minimalista senza extraction logic."""
    
    def __init__(self):
        print("🔧 Inizializzazione Simple RAG...")
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Carica vectorstore
        faiss_path = BASE_DIR / "data" / "vector_store" / "faiss_index"
        self.vectorstore = FAISS.load_local(
            str(faiss_path),
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        print("✅ Simple RAG pronto\n")
    
    def answer(self, question: str, k: int = 5) -> Dict:
        """
        Risponde usando solo retrieval + LLM.
        Zero extraction logic.
        """
        start_time = time.time()
        
        # 1. Retrieval
        docs = self.vectorstore.similarity_search(question, k=k)
        
        # 2. Format context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            filename = meta.get("filename", "Unknown")
            page = meta.get("page_number", "?")
            
            context_parts.append(f"[Doc {i}] {filename} (pag. {page})")
            context_parts.append(doc.page_content)
            context_parts.append("")  # Riga vuota
        
        context = "\n".join(context_parts)
        
        # 3. System Prompt (SEMPLICE)
        system_prompt = """Sei PALES-AI, assistente per ClassyFarm.

REGOLE:
1. Rispondi SOLO usando il CONTESTO fornito sotto
2. Se la risposta è nel contesto, rispondi in modo chiaro e completo
3. Se NON è nel contesto, rispondi: "Non ho informazioni su questo nei documenti disponibili"
4. Cita sempre le fonti: [Doc X]
5. Sii conciso ma completo

Non inventare informazioni. Non usare conoscenza esterna."""

        user_prompt = f"""{system_prompt}

=== CONTESTO ===
{context}

=== DOMANDA ===
{question}

=== RISPOSTA ==="""
        
        # 4. Generate
        response = self.llm.invoke(user_prompt)
        answer = response.content.strip()
        
        # 5. Format sources
        sources = []
        for doc in docs:
            meta = doc.metadata or {}
            sources.append({
                "filename": meta.get("filename", "Unknown"),
                "page_number": meta.get("page_number"),
            })
        
        latency = round(time.time() - start_time, 3)
        
        return {
            "answer": answer,
            "sources": sources,
            "latency": latency,
            "docs_retrieved": k,
            "method": "simple_rag"
        }


# ============================================
# TEST SUITE
# ============================================

def run_comparison_test():
    """
    Confronta Simple RAG vs Sistema Attuale su domande critiche.
    """
    
    # Domande test (le peggiori dal RAGAS)
    test_questions = [
        {
            "id": "Q010",
            "question": "Cosa si intende per OdC?",
            "issue": "Answer Relevancy 0.0",
            "expected_keywords": ["organismo", "certificazione"]
        },
        {
            "id": "Q014",
            "question": "Cosa vogliono dire Q1, Q2, Q3, Q4 nei cruscotti?",
            "issue": "Answer Relevancy 0.0",
            "expected_keywords": ["quartili", "percentili", "soglie"]
        },
        {
            "id": "Q021",
            "question": "Per cosa sta BDN?",
            "issue": "Answer Correctness 0.45",
            "expected_keywords": ["banca", "dati", "nazionale"]
        },
        {
            "id": "Q028",
            "question": "Quanto tempo dovrò attendere per l'abilitazione?",
            "issue": "Faithfulness 0.33",
            "expected_keywords": ["giorni", "ore", "tempo"]
        },
        {
            "id": "Q009",
            "question": "Come faccio a scaricare il report SQNBA?",
            "issue": "Answer Correctness 0.16",
            "expected_keywords": ["scaricare", "report", "esporta"]
        },
    ]
    
    # Init systems
    print("="*80)
    print("🧪 TEST EMPIRICO: Simple RAG vs Sistema Attuale")
    print("="*80)
    print()
    
    simple_rag = SimpleRAG()
    
    print("🔧 Sistema Attuale già inizializzato")
    print()
    
    # Results storage
    results = {
        "simple": [],
        "current": []
    }
    
    # Run tests
    for test in test_questions:
        print("="*80)
        print(f"[{test['id']}] {test['question']}")
        print(f"Issue noto: {test['issue']}")
        print("="*80)
        
        # ===== SIMPLE RAG =====
        print("\n🟢 SIMPLE RAG")
        print("-"*80)
        
        try:
            simple_result = simple_rag.answer(test['question'], k=5)
            
            answer_simple = simple_result['answer']
            latency_simple = simple_result['latency']
            sources_simple = len(simple_result['sources'])
            
            # Check keywords
            answer_lower = answer_simple.lower()
            keywords_found = sum(1 for kw in test['expected_keywords'] if kw in answer_lower)
            keyword_coverage = keywords_found / len(test['expected_keywords'])
            
            print(f"⏱️  Latenza: {latency_simple}s")
            print(f"📚 Documenti usati: {sources_simple}")
            print(f"🔑 Keyword coverage: {keywords_found}/{len(test['expected_keywords'])} ({keyword_coverage:.0%})")
            print(f"\n📝 Risposta:")
            print(f"{answer_simple[:300]}...")
            
            results['simple'].append({
                "id": test['id'],
                "latency": latency_simple,
                "answer_length": len(answer_simple),
                "keyword_coverage": keyword_coverage,
                "sources": sources_simple
            })
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            results['simple'].append({"id": test['id'], "error": str(e)})
        
        # ===== SISTEMA ATTUALE =====
        print("\n🔵 SISTEMA ATTUALE")
        print("-"*80)
        
        try:
            current_result = process_request(test['question'], chat_history=[])
            
            answer_current = current_result.get('answer', '')
            latency_current = current_result.get('latency', 0)
            sources_current = len(current_result.get('sources', []))
            mode_current = current_result.get('answer_mode', 'unknown')
            
            # Check keywords
            answer_lower = answer_current.lower()
            keywords_found = sum(1 for kw in test['expected_keywords'] if kw in answer_lower)
            keyword_coverage = keywords_found / len(test['expected_keywords'])
            
            print(f"⏱️  Latenza: {latency_current}s")
            print(f"📚 Documenti usati: {sources_current}")
            print(f"🎯 Modalità: {mode_current}")
            print(f"🔑 Keyword coverage: {keywords_found}/{len(test['expected_keywords'])} ({keyword_coverage:.0%})")
            print(f"\n📝 Risposta:")
            print(f"{answer_current[:300]}...")
            
            results['current'].append({
                "id": test['id'],
                "latency": latency_current,
                "answer_length": len(answer_current),
                "keyword_coverage": keyword_coverage,
                "sources": sources_current,
                "mode": mode_current
            })
            
        except Exception as e:
            print(f"❌ Errore: {e}")
            results['current'].append({"id": test['id'], "error": str(e)})
        
        print()
    
    # ===== SUMMARY =====
    print("="*80)
    print("📊 SUMMARY COMPARATIVO")
    print("="*80)
    
    # Calculate averages
    simple_results = [r for r in results['simple'] if 'error' not in r]
    current_results = [r for r in results['current'] if 'error' not in r]
    
    if simple_results and current_results:
        avg_latency_simple = sum(r['latency'] for r in simple_results) / len(simple_results)
        avg_latency_current = sum(r['latency'] for r in current_results) / len(current_results)
        
        avg_keyword_simple = sum(r['keyword_coverage'] for r in simple_results) / len(simple_results)
        avg_keyword_current = sum(r['keyword_coverage'] for r in current_results) / len(current_results)
        
        avg_length_simple = sum(r['answer_length'] for r in simple_results) / len(simple_results)
        avg_length_current = sum(r['answer_length'] for r in current_results) / len(current_results)
        
        print("\n🟢 SIMPLE RAG:")
        print(f"  • Latenza media: {avg_latency_simple:.2f}s")
        print(f"  • Keyword coverage: {avg_keyword_simple:.0%}")
        print(f"  • Lunghezza risposta: {avg_length_simple:.0f} chars")
        
        print("\n🔵 SISTEMA ATTUALE:")
        print(f"  • Latenza media: {avg_latency_current:.2f}s")
        print(f"  • Keyword coverage: {avg_keyword_current:.0%}")
        print(f"  • Lunghezza risposta: {avg_length_current:.0f} chars")
        
        print("\n📈 DELTA (Simple vs Current):")
        latency_delta = ((avg_latency_simple - avg_latency_current) / avg_latency_current) * 100
        keyword_delta = ((avg_keyword_simple - avg_keyword_current) / max(avg_keyword_current, 0.01)) * 100
        
        print(f"  • Latenza: {latency_delta:+.1f}%")
        print(f"  • Keyword coverage: {keyword_delta:+.1f}%")
        
        # Winner
        print("\n🏆 VINCITORE:")
        if avg_keyword_simple > avg_keyword_current * 1.1:  # 10% migliore
            print("  🟢 SIMPLE RAG - Risponde meglio con meno complessità")
        elif avg_keyword_current > avg_keyword_simple * 1.1:
            print("  🔵 SISTEMA ATTUALE - La complessità è giustificata")
        else:
            print("  🟡 PAREGGIO - Ma Simple RAG è più semplice da mantenere")
    
    print("\n" + "="*80)
    
    return results


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    try:
        results = run_comparison_test()
        
        # Salva risultati
        import json
        output_file = BASE_DIR / "results" / "simple_rag_comparison.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Risultati salvati in: {output_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrotto dall'utente")
    except Exception as e:
        print(f"\n\n❌ Errore durante il test: {e}")
        import traceback
        traceback.print_exc()