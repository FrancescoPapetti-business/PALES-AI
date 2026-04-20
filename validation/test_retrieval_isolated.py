"""
Test retrieval isolato per diagnosticare problemi di ranking.

Usage:
  python validation/test_retrieval_isolated.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
import re

# Setup
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Carica variabili ambiente
env_path = project_root / ".env"
load_dotenv(env_path)
print(f"✅ Loaded .env from: {env_path}\n")

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from core.layers.layer7_vector_store import load_bm25_retriever


# ===========================
# PREPROCESSING FUNCTION
# ===========================
def preprocess_for_bm25(text: str) -> str:
    """
    Preprocessing per BM25:
    - Lowercase
    - Rimuovi TUTTI gli apostrofi/virgolette
    - Rimuovi punteggiatura
    - Normalizza spazi
    """
    if not text:
        return ""
    
    text = text.lower()
    
    # Rimuovi apostrofi e virgolette (tutti i tipi)
    apostrophes = "''`ʼʻ'\"„""«»"
    for char in apostrophes:
        text = text.replace(char, '')
    
    # Rimuovi punteggiatura
    punctuation = '.,;:!?()[]{}—–-'
    for char in punctuation:
        text = text.replace(char, ' ')
    
    # Normalizza spazi
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ===========================
# TEST CONFIGURATION
# ===========================
query = "qual'è la definizione di delegato?"
expected_chunk_id = "73a94cb9-98f0-47f3-8e34-ec04e186511b"


# ==========================
# TEST 1: FAISS PURE
# ==========================
print("="*80)
print("🔍 LOADING FAISS INDEX...")
print("="*80)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.load_local(
    str(project_root / "data/vector_store/faiss_index"),
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ FAISS loaded\n")

print("="*80)
print(f"🔍 FAISS PURE RETRIEVAL")
print(f"Query: {query}")
print("="*80)

# Recupera top 20
results = vector_store.similarity_search_with_score(query, k=20)

expected_found = False
expected_rank = None
expected_score = None

for i, (doc, score) in enumerate(results, 1):
    meta = doc.metadata or {}
    filename = meta.get("filename", "unknown")
    page = meta.get("page_number", "?")
    chunk_id = meta.get("chunk_id", "")
    
    is_expected = expected_chunk_id in chunk_id
    
    marker = "⭐ EXPECTED!" if is_expected else ""
    
    print(f"\n[{i}] Score: {score:.4f} {marker}")
    print(f"    File: {filename} (page {page})")
    print(f"    Chunk ID: {chunk_id[:20]}...")
    print(f"    Canonical: {meta.get('canonical_concepts', [])}")
    print(f"    Keywords: {meta.get('keywords', [])}")
    print(f"    Content preview:")
    print(f"    {doc.page_content[:150]}...")
    
    if is_expected:
        expected_found = True
        expected_rank = i
        expected_score = score

print("\n" + "="*80)
print("📊 FAISS SUMMARY")
print("="*80)

if expected_found:
    print(f"✅ Expected chunk FOUND at rank: {expected_rank}")
    print(f"   Score: {expected_score:.4f}")
    print(f"   Top 1 score: {results[0][1]:.4f}")
    print(f"   Gap from top 1: {expected_score - results[0][1]:.4f}")
else:
    print(f"❌ Expected chunk NOT FOUND in top 20")
    print(f"   This means FAISS embeddings failed to match semantically")


# ==========================
# TEST 2: BM25 PURE
# ==========================
print("\n" + "="*80)
print("🔍 BM25 PURE RETRIEVAL")
print("="*80)


bm25 = load_bm25_retriever()

bm25_found = False
bm25_rank = None

if bm25:
    bm25.k = 20
    
    # Preprocessa query
    preprocessed_query = preprocess_for_bm25(query)
    print(f"Query original:      '{query}'")
    print(f"Query preprocessed:  '{preprocessed_query}'\n")
    
    bm25_results = bm25.invoke(preprocessed_query)
    
    for i, doc in enumerate(bm25_results, 1):
        meta = doc.metadata or {}
        chunk_id = meta.get("chunk_id", "")
        filename = meta.get("filename", "unknown")
        
        is_expected = expected_chunk_id in chunk_id
        marker = "⭐ EXPECTED!" if is_expected else ""
        
        print(f"\n[{i}] {marker}")
        print(f"    File: {filename}")
        print(f"    Chunk ID: {chunk_id[:20]}...")
        print(f"    Keywords: {meta.get('keywords', [])}")
        print(f"    Content: {doc.page_content[:150]}...")
        
        if is_expected:
            bm25_found = True
            bm25_rank = i
    
    print("\n" + "="*80)
    print("📊 BM25 SUMMARY")
    print("="*80)
    
    if bm25_found:
        print(f"✅ Expected chunk FOUND at rank: {bm25_rank}")
    else:
        print(f"❌ Expected chunk NOT FOUND in top 20")
        print(f"   This means BM25 tokenization/scoring failed")
else:
    print("⚠️ BM25 retriever not available")
print("\n🔍 Top 5 BM25 results:")
for i, doc in enumerate(bm25_results[:5], 1):
    meta = doc.metadata or {}
    content_preview = doc.page_content[:100].replace('\n', ' ')
    keyword_count = doc.page_content.lower().count('delegato')
    
    print(f"\n[{i}]")
    print(f"    File: {meta.get('filename', 'unknown')}")
    print(f"    'delegato' count: {keyword_count}")
    print(f"    Content: {content_preview}...")


# ==========================
# TEST 3: BM25 DIAGNOSTICS
# ==========================
print("\n" + "="*80)
print("🔬 BM25 DEEP DIAGNOSTICS")
print("="*80)

if bm25:
    # 1. Quanti documenti totali nell'indice?
    print(f"\n📊 BM25 Index Stats:")
    print(f"   Total documents indexed: {len(bm25.docs) if hasattr(bm25, 'docs') else 'N/A'}")
    
    # 2. Cerca manualmente il chunk atteso
    print(f"\n🔍 Manual search for expected chunk...")
    
    expected_found_manual = False
    if hasattr(bm25, 'docs'):
        for i, doc in enumerate(bm25.docs):
            chunk_id = doc.metadata.get("chunk_id", "")
            if expected_chunk_id in chunk_id:
                expected_found_manual = True
                print(f"   ✅ FOUND at index position: {i}")
                print(f"   Content: {doc.page_content[:150]}...")
                
                # 3. Verifica tokenizzazione
                print(f"\n🔤 Tokenization check:")
                query_tokens = query.lower().split()
                content_tokens = doc.page_content.lower().split()
                
                print(f"   Query tokens (original): {query_tokens}")
                
                preprocessed_query_tokens = preprocess_for_bm25(query).split()
                print(f"   Query tokens (preprocessed): {preprocessed_query_tokens}")
                
                print(f"   Chunk tokens (first 20): {content_tokens[:20]}")
                
                # 4. Keyword overlap
                common_orig = set(query_tokens) & set(content_tokens)
                common_prep = set(preprocessed_query_tokens) & set(content_tokens)
                
                print(f"   Common tokens (original): {common_orig}")
                print(f"   Common tokens (preprocessed): {common_prep}")
                
                if "delegato" in preprocessed_query_tokens and "delegato" in content_tokens:
                    print(f"   ✅ 'delegato' IS in both (preprocessed)")
                else:
                    print(f"   ❌ 'delegato' NOT matching")
                
                break
    
    if not expected_found_manual:
        print(f"   ❌ Chunk NOT in BM25 index at all!")
        print(f"   → The chunk was not indexed by BM25")
        print(f"   → Check layer7 BM25 creation logic")

print("="*80)


# ==========================
# FINAL DIAGNOSIS
# ==========================
print("\n" + "="*80)
print("🎯 DIAGNOSIS")
print("="*80)

if expected_found and bm25_found:
    print("✅ Both FAISS and BM25 found the chunk")
    print(f"   FAISS rank: {expected_rank}, BM25 rank: {bm25_rank}")
    print(f"   → Problem is in ENSEMBLE MERGING/RERANKING")
elif expected_found and not bm25_found:
    print("⚠️ Only FAISS found the chunk")
    print(f"   FAISS rank: {expected_rank}")
    print(f"   → BM25 is failing (tokenization/scoring issue)")
elif not expected_found and bm25_found:
    print("⚠️ Only BM25 found the chunk")
    print(f"   BM25 rank: {bm25_rank}")
    print(f"   → FAISS embeddings are failing")
else:
    print("❌ NEITHER FAISS NOR BM25 found the chunk")
    print(f"   → Critical retrieval failure")
    print(f"   → Check if chunk exists in indices")

print("="*80)