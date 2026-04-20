"""
update_index.py
Script di utilità per l'aggiornamento MANUALE dell'indice FAISS.
Carica tutti i file .txt presenti in 'data/data_rag_ready' e li aggiunge all'indice esistente.

NOTA: Per evitare duplicati in produzione, utilizzare 'orchestrator.py'
che gestisce il versioning tramite manifest.json.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================
#   SETUP PERCORSI
# ==========================
# Risolve la root 'classyfarm-rag'
# tools/update_index.py -> parents[1] -> root
BASE_DIR = Path(__file__).resolve().parents[1]

# Carica variabili d'ambiente da utils/.env
ENV_PATH = BASE_DIR / "utils" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Cartelle Dati
DATA_DIR = BASE_DIR / "data" / "data_rag_ready"
INDEX_DIR = BASE_DIR / "data" / "faiss_index"


def main():
    print("🔧 Avvio aggiornamento manuale indice FAISS...")
    
    # 0. Controlli preliminari
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Errore: OPENAI_API_KEY non trovata in utils/.env")
        return

    if not INDEX_DIR.exists() or not (INDEX_DIR / "index.faiss").exists():
        print(f"❌ Errore: Indice non trovato in {INDEX_DIR}.")
        print("   Esegui prima l'ingestion completa (orchestrator.py full).")
        return

    # 1. Carica indice esistente
    print(f"📂 Caricamento indice da: {INDEX_DIR}")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    try:
        db = FAISS.load_local(
            str(INDEX_DIR), 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"❌ Errore caricamento indice: {e}")
        return

    # 2. Carica i file dalla cartella RAG Ready
    print(f"📂 Scansione documenti in: {DATA_DIR}")
    docs = []
    
    if not DATA_DIR.exists():
        print("⚠ Directory dati non trovata.")
        return

    for path in DATA_DIR.glob("*.txt"):
        try:
            # Qui carichiamo tutto quello che c'è nella cartella.
            # ATTENZIONE: Se il file è già nell'indice, verrà duplicato.
            # Questo script è pensato per aggiornamenti rapidi o test.
            loader = TextLoader(str(path), encoding="utf-8")
            loaded_docs = loader.load()
            if loaded_docs:
                docs.append(loaded_docs[0])
        except Exception as e:
            print(f"⚠ Impossibile leggere {path.name}: {e}")

    if not docs:
        print("⚠ Nessun documento .txt trovato da aggiungere.")
        return

    print(f"   → Trovati {len(docs)} documenti.")

    # 3. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    new_chunks = splitter.split_documents(docs)
    print(f"   → Generati {len(new_chunks)} nuovi chunk.")

    # 4. Aggiorna FAISS
    print("🧠 Aggiunta chunk all'indice vettoriale...")
    db.add_documents(new_chunks)

    # 5. Salva il nuovo indice aggiornato
    db.save_local(str(INDEX_DIR))
    print(f"✅ Indice salvato correttamente in {INDEX_DIR}!")


if __name__ == "__main__":
    main()