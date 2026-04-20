"""
Debug script per tracciare il flusso di una query attraverso la pipeline.

Usage:
  python validation/debug_query.py --question "Come associare allevamento?"
"""

import sys
from pathlib import Path
import json

# Setup path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from core.layers.layer8_application import RAGEngine, process_request
from core.layers.layer4_semantic_rewriting import normalize_concepts

def trace_query(question: str, role: str = "operatore"):
    """
    Esegue una query e traccia ogni step della pipeline.
    """
    print("="*80)
    print(f"🔍 TRACING QUERY: {question}")
    print("="*80)
    
    # STEP 1: Query Normalization
    print("\n📍 STEP 1: Query Normalization")
    norm = normalize_concepts(question)
    print(f"   Canonical concepts found: {norm.get('canonical_concepts', [])}")
    print(f"   Normalized text: {norm.get('normalized_text', '')[:100]}...")
    
    # STEP 2: Process Request (con monkey-patch per intercettare dati)
    print("\n📍 STEP 2-13: Full Pipeline Execution")
    
    # Intercetta l'engine per accedere ai retriever
    engine = RAGEngine()
    
    # Hook nel retrieval (modifica temporanea)
    original_retrieve = engine.main_retriever._get_relevant_documents
    retrieved_docs = []
    
    def hooked_retrieve(query, **kwargs):
        docs = original_retrieve(query, **kwargs)
        retrieved_docs.extend(docs)
        print(f"\n   📚 Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs[:3], 1):
            meta = doc.metadata or {}
            print(f"      [{i}] {meta.get('filename', 'unknown')} (page {meta.get('page_number', '?')})")
            print(f"          Category: {meta.get('chunk_category', 'N/A')}")
            print(f"          Canonical: {meta.get('canonical_concepts', [])}")
            print(f"          Preview: {doc.page_content[:80]}...")
        return docs
    
    engine.main_retriever._get_relevant_documents = hooked_retrieve
    
    # Esegui la query
    result = process_request(question=question, role=role)
    
    # Restore
    engine.main_retriever._get_relevant_documents = original_retrieve
    
    # STEP 3: Analizza risultati
    print("\n📍 STEP 14: Final Result Analysis")
    print(f"   Answer length: {len(result.get('answer', ''))} chars")
    print(f"   Sources found: {len(result.get('sources', []))}")
    print(f"   Contexts retrieved: {len(result.get('retrieved_contexts', []))}")
    print(f"   Latency: {result.get('latency', 0):.2f}s")
    print(f"   Safety flags: {result.get('safety_flags', {})}")
    
    print("\n📄 ANSWER:")
    print(f"   {result.get('answer', 'N/A')[:300]}...")
    
    print("\n💾 CONTEXTS (first 2):")
    for i, ctx in enumerate(result.get('retrieved_contexts', [])[:2], 1):
        print(f"   [{i}] {ctx[:150]}...")
    
    # STEP 4: Export dettagli
    debug_data = {
        "question": question,
        "normalized": norm,
        "retrieved_docs_count": len(retrieved_docs),
        "retrieved_docs_metadata": [
            {
                "filename": d.metadata.get("filename"),
                "page": d.metadata.get("page_number"),
                "category": d.metadata.get("chunk_category"),
                "canonical": d.metadata.get("canonical_concepts"),
                "preview": d.page_content[:200]
            }
            for d in retrieved_docs
        ],
        "result": {
            "answer": result.get("answer"),
            "sources": result.get("sources"),
            "latency": result.get("latency")
        }
    }
    
    output_file = project_root / "validation" / f"debug_{question[:30].replace(' ', '_')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Debug data saved to: {output_file}")
    print("="*80)
    
    return result, debug_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug query pipeline")
    parser.add_argument("--question", type=str, required=True, help="Question to debug")
    parser.add_argument("--role", type=str, default="operatore", help="User role")
    
    args = parser.parse_args()
    
    result, debug_data = trace_query(args.question, args.role)