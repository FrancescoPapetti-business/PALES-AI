"""Verifica DOVE si trova il documento giusto nel ranking."""

import sys
from pathlib import Path
from dotenv import load_dotenv  # ← AGGIUNTO

# Setup path e carica .env
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ← CARICA .ENV PRIMA DI IMPORTARE OPENAI
load_dotenv(BASE_DIR / ".env")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

faiss_path = BASE_DIR / "data" / "vector_store" / "faiss_index"
vectorstore = FAISS.load_local(
    str(faiss_path),
    embeddings,
    allow_dangerous_deserialization=True
)

# Test domande critiche
tests = [
    {
        "question": "Cosa si intende per OdC?",
        "expected_content": "organismo di certificazione"
    },
    {
        "question": "Cosa vogliono dire Q1, Q2, Q3, Q4 nei cruscotti?",
        "expected_content": "quartili"
    },
    {
        "question": "Quanto tempo dovrò attendere per l'abilitazione?",
        "expected_content": ["48 ore", "giorni", "entro"]
    },
    {
        "question": "Per cosa sta BDN?",
        "expected_content": "banca dati nazionale"
    },
    {
        "question": "Come faccio a scaricare il report SQNBA?",
        "expected_content": ["scaricare", "report", "esporta"]
    },
]

print("\n" + "="*80)
print("🔍 ANALISI QUALITÀ RETRIEVAL - Posizione Documenti Rilevanti")
print("="*80)

summary = []

for test in tests:
    print(f"\n{'='*80}")
    print(f"Q: {test['question']}")
    print('='*80)
    
    # Recupera top-20 per analisi
    docs = vectorstore.similarity_search(test['question'], k=20)
    
    expected = test['expected_content']
    if not isinstance(expected, list):
        expected = [expected]
    
    # Trova posizione documento rilevante
    found = False
    for i, doc in enumerate(docs, 1):
        content_lower = doc.page_content.lower()
        
        if any(exp.lower() in content_lower for exp in expected):
            print(f"\n✅ Documento rilevante trovato in posizione #{i}")
            print(f"   Filename: {doc.metadata.get('filename')}")
            print(f"   Page: {doc.metadata.get('page_number')}")
            print(f"   Snippet: {doc.page_content[:150]}...")
            
            if i <= 5:
                status = "🟢 OK (top-5)"
            elif i <= 10:
                status = "🟡 BORDERLINE (top-10)"
            else:
                status = "🔴 PROBLEMA (fuori top-10)"
            
            print(f"\n   Status: {status}")
            
            summary.append({
                "question": test['question'][:50],
                "position": i,
                "status": status
            })
            
            found = True
            break
    
    if not found:
        print(f"\n❌ Documento rilevante NON trovato nei top-20")
        summary.append({
            "question": test['question'][:50],
            "position": ">20",
            "status": "🔴 CRITICO (non trovato)"
        })

# Summary
print("\n" + "="*80)
print("📊 SUMMARY POSIZIONI")
print("="*80)

for item in summary:
    print(f"{item['status']} Pos #{item['position']:>3} | {item['question']}")

# Statistics
positions = [s['position'] for s in summary if isinstance(s['position'], int)]
if positions:
    avg_pos = sum(positions) / len(positions)
    print(f"\n📈 Posizione media: {avg_pos:.1f}")
    
    in_top5 = sum(1 for p in positions if p <= 5)
    in_top10 = sum(1 for p in positions if p <= 10)
    
    print(f"   • In top-5: {in_top5}/{len(tests)} ({in_top5/len(tests)*100:.0f}%)")
    print(f"   • In top-10: {in_top10}/{len(tests)} ({in_top10/len(tests)*100:.0f}%)")

print("\n" + "="*80)