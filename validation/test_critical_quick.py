"""
Test rapido domande critiche - NO REBUILD REQUIRED
"""
import sys
from pathlib import Path

# Aggiungi la root del progetto al PYTHONPATH
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import corretto: process_request è una funzione, non un metodo
from core.layers.layer8_application import RAGEngine, process_request

CRITICAL_TESTS = {
    "Q010": "Cosa si intende per OdC?",
    "Q014": "Cosa vogliono dire Q1, Q2, Q3, Q4 nei cruscotti?",
    "Q021": "Per cosa sta BDN?",
    "Q028": "Quanto tempo dovrò attendere per l'abilitazione?",
    "Q009": "Come faccio a scaricare il report SQNBA?",
}

print("\n" + "="*80)
print("🧪 TEST DOMANDE CRITICHE - POST MODIFICHE (NO REBUILD)")
print("="*80)

# Inizializza il RAG Engine (singleton)
try:
    engine = RAGEngine()
    print("✅ RAG Engine inizializzato correttamente\n")
except Exception as e:
    print(f"❌ Errore inizializzazione: {e}")
    sys.exit(1)

results = {}

for qid, question in CRITICAL_TESTS.items():
    print(f"\n{'─'*80}")
    print(f"📋 [{qid}]")
    print(f"❓ Domanda: {question}")
    print('─'*80)
    
    try:
        # USA process_request DIRETTAMENTE (non engine.process_request)
        response = process_request(question, chat_history=[])
        
        answer = response.get('answer', '')
        sources = response.get('sources', [])
        latency = response.get('latency', 0)
        bm25_active = response.get('bm25_active', False)
        
        # Analisi risposta
        has_temporal = any(word in answer for word in ["ore", "giorni", "settimane", "⏰"])
        has_acronym_expanded = any(word in answer.lower() for word in 
            ["organismo", "banca dati", "sistema qualità", "certificazione"])
        answer_length = len(answer)
        
        print(f"\n📝 Risposta ({answer_length} chars):")
        print(f"   {answer[:300]}{'...' if len(answer) > 300 else ''}")
        
        print(f"\n📊 Metriche:")
        print(f"   • Fonti recuperate: {len(sources)}")
        print(f"   • Latenza: {latency:.2f}s")
        print(f"   • BM25 attivo: {'✅' if bm25_active else '❌'}")
        print(f"   • Contiene valori temporali: {'✅' if has_temporal else '❌'}")
        print(f"   • Acronimi espansi: {'✅' if has_acronym_expanded else '❌'}")
        
        if sources:
            print(f"\n📚 Fonti (top 3):")
            for i, src in enumerate(sources[:3], 1):
                fname = src.get('filename', 'Unknown')
                page = src.get('page_number', 'N/A')
                print(f"   [{i}] {fname} (pag. {page})")
        
        results[qid] = {
            "success": len(answer) > 50 and not answer.startswith("Non ho trovato"),
            "has_temporal": has_temporal,
            "has_acronym": has_acronym_expanded,
            "answer_length": answer_length,
            "latency": latency
        }
        
    except Exception as e:
        print(f"\n❌ ERRORE durante elaborazione: {e}")
        import traceback
        traceback.print_exc()
        results[qid] = {"success": False, "error": str(e)}

# Summary
print("\n" + "="*80)
print("📊 RIEPILOGO RISULTATI")
print("="*80)

success_count = sum(1 for r in results.values() if r.get("success", False))
temporal_count = sum(1 for r in results.values() if r.get("has_temporal", False))
acronym_count = sum(1 for r in results.values() if r.get("has_acronym", False))

print(f"\n✅ Risposte valide: {success_count}/{len(CRITICAL_TESTS)}")
print(f"⏰ Con valori temporali: {temporal_count}")
print(f"🔤 Con espansione acronimi: {acronym_count}")

if success_count > 0:
    avg_latency = sum(r.get("latency", 0) for r in results.values() if r.get("success")) / success_count
    print(f"⏱️  Latenza media: {avg_latency:.2f}s")

print("\n📋 Dettaglio per domanda:")
for qid, result in results.items():
    status = "✅" if result.get("success") else "❌"
    print(f"  {status} {qid}: {'OK' if result.get('success') else result.get('error', 'FAIL')}")

print("\n" + "="*80 + "\n")