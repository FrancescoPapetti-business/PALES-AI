"""Test Query Rewriting per migliorare retrieval."""

import sys
from pathlib import Path
from dotenv import load_dotenv  # ← AGGIUNTO

# Setup
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# ← CARICA .ENV
load_dotenv(BASE_DIR / ".env")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

faiss_path = BASE_DIR / "data" / "vector_store" / "faiss_index"
vectorstore = FAISS.load_local(
    str(faiss_path), 
    embeddings, 
    allow_dangerous_deserialization=True
)

def rewrite_for_search(question: str) -> str:
    """Riscrive la query per retrieval migliore."""
    
    prompt = f"""Riformula questa domanda per una ricerca semantica più efficace in un database di documenti tecnici.

OBIETTIVO: Creare una query che matchi meglio i documenti.

REGOLE:
1. Espandi acronimi se presenti (es. OdC → organismo certificazione)
2. Aggiungi termini tecnici correlati
3. Rimuovi parole di cortesia
4. Mantieni termini chiave
5. Output: SOLO la query ottimizzata, senza spiegazioni

Domanda originale: {question}

Query ottimizzata:"""
    
    response = llm.invoke(prompt)
    return response.content.strip()

# Test questions
questions = [
    "Cosa si intende per OdC?",
    "Cosa vogliono dire Q1, Q2, Q3, Q4 nei cruscotti?",
    "Quanto tempo dovrò attendere per l'abilitazione?",
    "Per cosa sta BDN?",
    "Come faccio a scaricare il report SQNBA?",
]

print("\n" + "="*80)
print("🔄 TEST QUERY REWRITING - Confronto Retrieval")
print("="*80)

improvements = []

for q in questions:
    print(f"\n{'='*80}")
    print(f"Originale: {q}")
    
    # Rewrite
    rewritten = rewrite_for_search(q)
    print(f"Riscritta: {rewritten}")
    print("-"*80)
    
    # Retrieval SENZA rewriting
    print("\n🔵 Top-3 SENZA rewriting:")
    docs_original = vectorstore.similarity_search(q, k=3)
    for i, doc in enumerate(docs_original, 1):
        filename = doc.metadata.get('filename', 'Unknown')
        snippet = doc.page_content[:100].replace('\n', ' ')
        print(f"  [{i}] {filename}")
        print(f"      {snippet}...")
    
    # Retrieval CON rewriting
    print("\n🟢 Top-3 CON rewriting:")
    docs_rewritten = vectorstore.similarity_search(rewritten, k=3)
    for i, doc in enumerate(docs_rewritten, 1):
        filename = doc.metadata.get('filename', 'Unknown')
        snippet = doc.page_content[:100].replace('\n', ' ')
        print(f"  [{i}] {filename}")
        print(f"      {snippet}...")
    
    # Compare
    original_files = [d.metadata.get('filename') for d in docs_original]
    rewritten_files = [d.metadata.get('filename') for d in docs_rewritten]
    
    if original_files != rewritten_files:
        print("\n⚠️  CAMBIO NEI DOCUMENTI RECUPERATI")
        improvements.append(q)
    else:
        print("\n➡️  Stessi documenti")

# Summary
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"\nQuery con retrieval DIVERSO dopo rewriting: {len(improvements)}/{len(questions)}")

if improvements:
    print("\nDomande migliorate:")
    for q in improvements:
        print(f"  • {q}")

print("\n" + "="*80)