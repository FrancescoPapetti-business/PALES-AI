# File: test_comparative_10q.py

"""
Test comparativo finale: 10 domande rappresentative
Esclude Q010 (OdC) come richiesto
"""

import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

from core.simple_rag_engine import process_request as simple_request
from core.layers.layer8_application import process_request as current_request

# ============================================
# 10 Domande Selezionate
# ============================================

QUESTIONS = [
    {"id": "Q003", "q": "Come richiedo il profilo da operatore?", "keywords": ["richiesta", "modulo", "info@classyfarm.it"]},
    {"id": "Q006", "q": "Come faccio ad associare un valutatore?", "keywords": ["selezione valutatore", "codice fiscale"]},
    {"id": "Q008", "q": "Come faccio a continuare una compilazione di checklist di valutazione salvata in bozza?", "keywords": ["bozza", "storico", "modifica"]},
    {"id": "Q009", "q": "Come faccio a scaricare il report SQNBA?", "keywords": ["report", "sqnba", "48 ore"]},
    {"id": "Q021", "q": "Per cosa sta BDN?", "keywords": ["banca dati nazionale"]},
    {"id": "Q012", "q": "Come faccio ad associarmi all'allevamento?", "keywords": ["associare", "allevamento", "selezione azienda"]},
    {"id": "Q014", "q": "Cosa vogliono dire Q1, Q2, Q3, Q4 nei cruscotti?", "keywords": ["quartili", "cruscotto"]},
    {"id": "Q017", "q": "Come faccio a modificare la data delle prescrizioni in un controllo già compilato?", "keywords": ["copia", "modifica", "controllo"]},
    {"id": "Q046", "q": "Come devono essere i pavimenti delle stalle dei suini?", "keywords": ["pavimenti", "antisdrucciolevoli", "suini"]},
    {"id": "Q050", "q": "Qual è la definizione di operatore ai sensi del Regolamento UE 429/2016?", "keywords": ["persona fisica", "giuridica", "responsabile"]},
]

print("\n" + "="*80)
print("⚔️  TEST COMPARATIVO: 10 Domande Rappresentative")
print("="*80)

results = {"simple": [], "current": []}

for test in QUESTIONS:
    qid = test["id"]
    question = test["q"]
    keywords = test["keywords"]
    
    print(f"\n{'─'*80}")
    print(f"[{qid}] {question}")
    print('─'*80)
    
    # ===== SIMPLE RAG =====
    print("\n🟢 SIMPLE RAG:")
    try:
        simple = simple_request(question)
        answer_s = simple['answer']
        latency_s = simple['latency']
        
        # Keyword coverage
        kw_found = sum(1 for kw in keywords if kw.lower() in answer_s.lower())
        kw_coverage_s = kw_found / len(keywords)
        
        print(f"   {answer_s[:120]}...")
        print(f"   ⏱️  {latency_s:.2f}s | 🔑 {kw_found}/{len(keywords)} keywords")
        
        results['simple'].append({
            'id': qid,
            'latency': latency_s,
            'answer_length': len(answer_s),
            'keyword_coverage': kw_coverage_s,
            'has_answer': len(answer_s) > 60 and "non ho informazioni" not in answer_s.lower()
        })
        
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        results['simple'].append({'id': qid, 'error': str(e)})
    
    # ===== SISTEMA ATTUALE =====
    print("\n🔵 SISTEMA ATTUALE:")
    try:
        current = current_request(question, chat_history=[])
        answer_c = current['answer']
        latency_c = current['latency']
        
        # Keyword coverage
        kw_found = sum(1 for kw in keywords if kw.lower() in answer_c.lower())
        kw_coverage_c = kw_found / len(keywords)
        
        print(f"   {answer_c[:120]}...")
        print(f"   ⏱️  {latency_c:.2f}s | 🔑 {kw_found}/{len(keywords)} keywords")
        
        results['current'].append({
            'id': qid,
            'latency': latency_c,
            'answer_length': len(answer_c),
            'keyword_coverage': kw_coverage_c,
            'has_answer': len(answer_c) > 60
        })
        
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        results['current'].append({'id': qid, 'error': str(e)})

# ===== SUMMARY =====
print("\n" + "="*80)
print("📊 SUMMARY FINALE")
print("="*80)

simple_ok = [r for r in results['simple'] if 'error' not in r]
current_ok = [r for r in results['current'] if 'error' not in r]

if simple_ok and current_ok:
    # Metriche
    avg_lat_s = sum(r['latency'] for r in simple_ok) / len(simple_ok)
    avg_lat_c = sum(r['latency'] for r in current_ok) / len(current_ok)
    
    avg_kw_s = sum(r['keyword_coverage'] for r in simple_ok) / len(simple_ok)
    avg_kw_c = sum(r['keyword_coverage'] for r in current_ok) / len(current_ok)
    
    valid_s = sum(1 for r in simple_ok if r['has_answer'])
    valid_c = sum(1 for r in current_ok if r['has_answer'])
    
    print("\n🟢 SIMPLE RAG:")
    print(f"  • Latenza media: {avg_lat_s:.2f}s")
    print(f"  • Keyword coverage: {avg_kw_s:.1%}")
    print(f"  • Risposte valide: {valid_s}/{len(QUESTIONS)}")
    
    print("\n🔵 SISTEMA ATTUALE:")
    print(f"  • Latenza media: {avg_lat_c:.2f}s")
    print(f"  • Keyword coverage: {avg_kw_c:.1%}")
    print(f"  • Risposte valide: {valid_c}/{len(QUESTIONS)}")
    
    print("\n📈 DELTA (Simple - Current):")
    lat_delta = ((avg_lat_s - avg_lat_c) / avg_lat_c) * 100
    kw_delta = ((avg_kw_s - avg_kw_c) / max(avg_kw_c, 0.01)) * 100
    
    print(f"  • Latenza: {lat_delta:+.1f}%")
    print(f"  • Keyword coverage: {kw_delta:+.1f}%")
    print(f"  • Risposte valide: {valid_s - valid_c:+d}")
    
    print("\n🏆 VERDETTO:")
    
    if valid_s >= valid_c - 1 and avg_kw_s >= avg_kw_c * 0.9:
        print("  ✅ SIMPLE RAG COMPETITIVO - Qualità simile, architettura più semplice")
    elif valid_c > valid_s + 2:
        print("  🔵 SISTEMA ATTUALE VINCE - Complessità giustificata dalla qualità")
    else:
        print("  🤝 PAREGGIO - Valuta trade-off complessità/qualità")

print("\n" + "="*80 + "\n")

# Salva risultati
import json
output = BASE_DIR / "results" / "comparative_10q_results.json"
output.parent.mkdir(exist_ok=True)

with open(output, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"✅ Risultati salvati in: {output}\n")