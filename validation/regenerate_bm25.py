# validation/regenerate_bm25.py

import sys
from pathlib import Path
import re
import pickle
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from core.layers.layer7_vector_store import load_latest_dataset
from core.layers.layer6_dataset_manager import get_dataset_fingerprint

print("="*80)
print("🔨 REGENERATING BM25 INDEX WITH PREPROCESSING")
print("="*80)


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
# STEP 1: LOAD DATASET
# ===========================
print("\n📂 Step 1: Loading dataset...")
docs = load_latest_dataset()
print(f"   ✅ Loaded {len(docs)} documents")

if not docs or len(docs) == 0:
    print("   ❌ ERROR: No documents loaded!")
    exit(1)


# ===========================
# STEP 2: EXTRACT & PREPROCESS
# ===========================
print("\n📝 Step 2: Extracting and preprocessing texts...")
texts = []
metadatas = []

for i, doc in enumerate(docs):
    if not hasattr(doc, 'page_content'):
        print(f"   ⚠️ Doc {i} missing page_content, skipping")
        continue
    
    # ✅ APPLICA PREPROCESSING
    text = preprocess_for_bm25(doc.page_content)
    
    if not text or not text.strip():
        print(f"   ⚠️ Doc {i} has empty text after preprocessing, skipping")
        continue
    
    texts.append(text)
    metadatas.append(doc.metadata or {})

print(f"   ✅ Extracted {len(texts)} valid texts")

if len(texts) == 0:
    print("   ❌ ERROR: No valid texts extracted!")
    exit(1)

# Sample output per verificare preprocessing
print(f"\n📄 Sample preprocessed text:")
print(f"   Original: {docs[0].page_content[:100]}...")
print(f"   Preprocessed: {texts[0][:100]}...")


# ===========================
# STEP 3: CREATE WRAPPER
# ===========================
print("\n💾 Step 3: Creating BM25 wrapper...")

bm25_wrapper = {
    "format_version": "2.0",
    "dataset_fingerprint": get_dataset_fingerprint(),
    "texts": texts,
    "metadatas": metadatas,
    "created_at": datetime.now().isoformat(),
    "total_docs": len(texts)
}

print(f"   Format version: {bm25_wrapper['format_version']}")
print(f"   Total texts: {len(bm25_wrapper['texts'])}")
print(f"   Fingerprint: {bm25_wrapper['dataset_fingerprint']}")


# ===========================
# STEP 4: SAVE
# ===========================
bm25_path = project_root / "data" / "vector_store" / "bm25_retriever.pkl"
bm25_path.parent.mkdir(parents=True, exist_ok=True)

print(f"\n💾 Step 4: Saving to {bm25_path}...")

with open(bm25_path, "wb") as f:
    pickle.dump(bm25_wrapper, f)

print(f"   ✅ Saved successfully")


# ===========================
# STEP 5: VERIFY
# ===========================
print("\n🔍 Step 5: Verification...")

with open(bm25_path, "rb") as f:
    verify = pickle.load(f)

verify_texts = verify.get("texts", [])
print(f"   Texts in file: {len(verify_texts)}")

if len(verify_texts) == len(texts):
    print(f"   ✅ Verification PASSED")
else:
    print(f"   ❌ Verification FAILED: expected {len(texts)}, got {len(verify_texts)}")


# ===========================
# STEP 6: TEST LOADING
# ===========================
print("\n🧪 Step 6: Testing load_bm25_retriever()...")

from core.layers.layer7_vector_store import load_bm25_retriever

bm25 = load_bm25_retriever()

if bm25 and hasattr(bm25, 'docs'):
    print(f"   ✅ BM25 loaded successfully")
    print(f"   Documents in retriever: {len(bm25.docs)}")
else:
    print(f"   ❌ BM25 loading failed or has no docs")

print("\n" + "="*80)
print("✅ REGENERATION COMPLETE")
print("="*80)