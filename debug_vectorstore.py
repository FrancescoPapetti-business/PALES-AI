import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fissiamo BASE_DIR automaticamente
current = Path(__file__).resolve()
for parent in current.parents:
    if (parent / ".env").exists():
        BASE_DIR = parent
        break
else:
    raise RuntimeError("Impossibile trovare .env")

sys.path.append(str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

print("\n=== DEBUG VECTOR STORE ===")
print(f"BASE_DIR → {BASE_DIR}")
print(f"ENV_KEY_PRESENT → {os.getenv('OPENAI_API_KEY') is not None}")

# Import ufficiali
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_PATH = BASE_DIR / "data" / "vector_store" / "faiss_index"

print(f"\nCaricamento FAISS da: {FAISS_PATH}")

if not FAISS_PATH.exists():
    raise RuntimeError("❌ FAISS Index NON esiste nel path previsto!")

emb = OpenAIEmbeddings(model="text-embedding-3-small")

try:
    vector_store = FAISS.load_local(
        str(FAISS_PATH),
        emb,
        allow_dangerous_deserialization=True
    )
    print("✔ FAISS caricato con successo")

except Exception as e:
    print(f"❌ Errore caricamento FAISS: {e}")
    raise e

# Numero documenti
docs_count = len(vector_store.docstore._dict)
print(f"\n📚 Documenti indicizzati: {docs_count}")

if docs_count == 0:
    raise RuntimeError("❌ FAISS vuoto! Nessun documento indicizzato.")

# Mostriamo i primi 3 documenti
print("\n=== METADATA PRIMI DOCUMENTI ===")
for i, (key, doc) in enumerate(vector_store.docstore._dict.items()):
    if i > 2:
        break
    print(f"\n--- Documento {i+1} ---")
    print("Page content (primi 200 char):", doc.page_content[:200])
    print("Metadata:", doc.metadata)

    # Verifica presenza campi richiesti
    for field in ["filename", "page_number"]:
        if field not in doc.metadata:
            print(f"⚠ WARNING: campo '{field}' mancante nei metadata!")
        else:
            print(f"✔ Campo '{field}':", doc.metadata[field])

# Test retrieval FAISS
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("\n=== TEST QUERY RETRIEVER ===")
try:
    results = retriever.get_relevant_documents("ciao")
    print(f"Documenti recuperati: {len(results)}")
    for d in results:
        print(" ->", d.metadata.get("filename"), "/", d.metadata.get("page_number"))
except Exception as e:
    print(f"❌ Errore retrieval: {e}")

# Test EnsembleRetriever se esiste
print("\n=== TEST ENSEMBLE RETRIEVER ===")
try:
    from core.custom.ensemble_retriever import EnsembleRetriever
    print("✔ EnsembleRetriever trovato")
except Exception as e:
    print("⚠ EnsembleRetriever NON trovato o errore:", e)
