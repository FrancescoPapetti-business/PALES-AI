"""
Debug Layer 7 dataset loading per capire perché un chunk manca.

Usage:
  python validation/debug_layer7_loading.py
"""

import sys
from pathlib import Path
import json

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

print("="*80)
print("🔍 DEBUGGING LAYER 7 DATASET LOADING")
print("="*80)

# Chunk ID da cercare
expected_chunk_id = "73a94cb9-98f0-47f3-8e34-ec04e186511b"

# Trova dataset
DATASETS_DIR = project_root / "data" / "datasets"
versions = [d for d in DATASETS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
latest_version = sorted(versions, key=lambda x: x.name, reverse=True)[0]
jsonl_path = latest_version / "chunks.jsonl"

print(f"\n📂 Loading from: {jsonl_path}")
print(f"   Dataset version: {latest_version.name}\n")

total_chunks = 0
filtered_chunks = 0
expected_found_in_jsonl = False
expected_filtered = False
filter_reason = None

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        
        data = json.loads(line)
        chunk_id = data.get("chunk_id", "")
        total_chunks += 1
        
        # Controlla se è il nostro chunk
        is_expected = expected_chunk_id in chunk_id
        
        if is_expected:
            expected_found_in_jsonl = True
            text_preview = data.get("text_content", "")[:200]
            print(f"⭐ EXPECTED CHUNK FOUND IN JSONL (line {total_chunks})")
            print(f"   Chunk ID: {chunk_id}")
            print(f"   Text: {text_preview}...")
            print(f"   Metadata: {json.dumps(data.get('metadata', {}), indent=2)}")
        
        # Simula i filtri di Layer 7
        text_content = data.get("text_content", "") or ""
        safe_md = data.get("metadata", {}) or {}
        
        # Filtro 1: Lunghezza
        if len(text_content) > 3000:
            filtered_chunks += 1
            if is_expected:
                expected_filtered = True
                filter_reason = f"Text too long ({len(text_content)} chars)"
            continue
        
        # Filtro 2: Category
        cat = safe_md.get("chunk_category")
        if cat in ["sanctions_table", "checklist"]:
            filtered_chunks += 1
            if is_expected:
                expected_filtered = True
                filter_reason = f"Category filtered: {cat}"
            continue

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)
print(f"Total chunks in JSONL: {total_chunks}")
print(f"Filtered by Layer 7: {filtered_chunks}")
print(f"Would be indexed: {total_chunks - filtered_chunks}")

if expected_found_in_jsonl:
    print(f"\n✅ Expected chunk IS in JSONL")
    if expected_filtered:
        print(f"❌ BUT it was FILTERED by Layer 7")
        print(f"   Reason: {filter_reason}")
    else:
        print(f"✅ AND it PASSED all Layer 7 filters")
        print(f"   → BM25 should have indexed it")
        print(f"   → Problem is in BM25Retriever.from_texts()")
else:
    print(f"\n❌ Expected chunk NOT in JSONL")
    print(f"   → Problem is in Layer 1-6 (ingestion)")
    print(f"   → You need to regenerate the dataset from scratch")

print("="*80)