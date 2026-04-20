"""Test completo Fix BM25"""

from pathlib import Path
from core.layers.layer7_vector_store import load_bm25_index
from core.layers.layer8_application import RAGEngine, process_request

print('='*60)
print('BM25 FIX - TEST COMPLETO')
print('='*60)

# Test 1: File esistente
print('\n[1/4] Verifica File BM25')
bm25_path = Path('data/vector_store/bm25_retriever.pkl')
file_ok = bm25_path.exists()
print(f'      Exists: {"✅" if file_ok else "❌"}')

if file_ok:
    size_mb = bm25_path.stat().st_size / 1024 / 1024
    print(f'      Size: {size_mb:.2f} MB')

# Test 2: Caricamento
print('\n[2/4] Caricamento BM25')
bm25 = load_bm25_index()
load_ok = bm25 is not None and hasattr(bm25, 'k')
print(f'      Result: {"✅" if load_ok else "❌"}')

if load_ok:
    print(f'      Type: {type(bm25).__name__}')
    print(f'      k: {bm25.k}')

# Test 3: Engine Init
print('\n[3/4] RAG Engine Init')
try:
    engine = RAGEngine()
    engine_ok = engine.bm25_active
    print(f'      BM25 active: {"✅" if engine_ok else "❌"}')
    print(f'      Retriever: {type(engine.retriever).__name__}')
    
    if hasattr(engine.retriever, 'retrievers'):
        print(f'      Ensemble: {len(engine.retriever.retrievers)} retrievers')
except Exception as e:
    engine_ok = False
    print(f'      ❌ Errore: {e}')
    import traceback
    traceback.print_exc()

# Test 4: Retrieval
print('\n[4/4] Test Retrieval')
try:
    result = process_request('Per cosa sta BDN?', [])
    answer_ok = len(result['answer']) > 40
    print(f'      Answer: {"✅" if answer_ok else "❌"}')
    print(f'      Length: {len(result["answer"])} chars')
    print(f'      Preview: {result["answer"][:80]}...')
except Exception as e:
    answer_ok = False
    print(f'      ❌ Errore: {e}')
    import traceback
    traceback.print_exc()

# Risultato finale
print('\n' + '='*60)
print('RISULTATO FINALE')
print('='*60)

all_ok = file_ok and load_ok and engine_ok and answer_ok

if all_ok:
    print('✅ FIX BM25 COMPLETATO CON SUCCESSO!\n')
    print('Prossimo step: Fix 2 (Intent Classification)')
else:
    print('⚠️  Fix parziale o problemi rilevati:\n')
    if not file_ok:
        print('   • File BM25 mancante')
    if not load_ok:
        print('   • Caricamento BM25 fallito')
    if not engine_ok:
        print('   • Engine BM25 non attivo')
    if not answer_ok:
        print('   • Retrieval non funzionante')

print('='*60)