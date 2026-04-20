"""Confronto finale: Sistema Attuale vs Simple RAG"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

# Import entrambi i sistemi
from core.simple_rag_engine import process_request as simple_request
from core.layers.layer8_application import process_request as current_request

questions = [
    "Cosa si intende per OdC?",
    "Cosa vogliono dire Q1, Q2, Q3, Q4 nei cruscotti?",
    "Per cosa sta BDN?",
    "Quanto tempo dovrò attendere per l'abilitazione?",
    "Come faccio a scaricare il report SQNBA?",
]

print("\n" + "="*80)
print("⚔️  CONFRONTO FINALE: Simple RAG vs Sistema Attuale")
print("="*80)

results = {"simple": [], "current": []}

for q in questions:
    print(f"\n{'─'*80}")
    print(f"❓ {q}")
    print('─'*80)
    
    # Simple RAG
    print("\n🟢 SIMPLE RAG:")
    try:
        simple = simple_request(q)
        answer_simple = simple['answer']
        latency_simple = simple['latency']
        docs_simple = simple['docs_retrieved']
        
        print(f"   {answer_simple[:150]}...")
        print(f"   ⏱️  {latency_simple}s | 📚 {docs_simple} docs")
        
        results['simple'].append({
            'question': q,
            'answer_length': len(answer_simple),
            'latency': latency_simple,
            'has_answer': len(answer_simple) > 60 and "non ho informazioni" not in answer_simple.lower()
        })
        
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        results['simple'].append({'question': q, 'error': str(e)})
    
    # Sistema Attuale
    print("\n🔵 SISTEMA ATTUALE:")
    try:
        current = current_request(q, chat_history=[])
        answer_current = current['answer']
        latency_current = current['latency']
        docs_current = len(current.get('sources', []))
        
        print(f"   {answer_current[:150]}...")
        print(f"   ⏱️  {latency_current}s | 📚 {docs_current} docs")
        
        results['current'].append({
            'question': q,
            'answer_length': len(answer_current),
            'latency': latency_current,
            'has_answer': len(answer_current) > 60 and "non ho informazioni" not in answer_current.lower()
        })
        
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        results['current'].append({'question': q, 'error': str(e)})

# Summary
print("\n" + "="*80)
print("📊 SUMMARY COMPARATIVO")
print("="*80)

simple_ok = [r for r in results['simple'] if 'error' not in r]
current_ok = [r for r in results['current'] if 'error' not in r]

if simple_ok and current_ok:
    # Latenza
    avg_latency_simple = sum(r['latency'] for r in simple_ok) / len(simple_ok)
    avg_latency_current = sum(r['latency'] for r in current_ok) / len(current_ok)
    
    # Risposte valide
    valid_simple = sum(1 for r in simple_ok if r['has_answer'])
    valid_current = sum(1 for r in current_ok if r['has_answer'])
    
    print("\n🟢 SIMPLE RAG:")
    print(f"  • Latenza media: {avg_latency_simple:.2f}s")
    print(f"  • Risposte valide: {valid_simple}/{len(questions)}")
    
    print("\n🔵 SISTEMA ATTUALE:")
    print(f"  • Latenza media: {avg_latency_current:.2f}s")
    print(f"  • Risposte valide: {valid_current}/{len(questions)}")
    
    print("\n📈 DELTA:")
    latency_gain = ((avg_latency_current - avg_latency_simple) / avg_latency_current) * 100
    print(f"  • Velocità: {latency_gain:+.1f}% (Simple)")
    print(f"  • Qualità: {valid_simple - valid_current:+d} risposte")
    
    print("\n🏆 VERDETTO:")
    if valid_simple >= valid_current and avg_latency_simple < avg_latency_current:
        print("  ✅ SIMPLE RAG VINCE - Stessa qualità, più veloce e più semplice")
    elif valid_simple > valid_current:
        print("  ✅ SIMPLE RAG VINCE - Qualità superiore")
    elif valid_current > valid_simple:
        print("  ⚠️  SISTEMA ATTUALE VINCE - Ma valuta se la complessità vale la differenza")
    else:
        print("  🤝 PAREGGIO - Simple RAG preferibile per semplicità")

print("\n" + "="*80 + "\n")