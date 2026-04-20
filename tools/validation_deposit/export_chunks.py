import os
from pathlib import Path

# --- CORREZIONE 1: Caricamento API Key ---
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv()) 
# --- FINE CORREZIONE 1 ---

from langchain_openai import OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS

def export_chunks(query: str, k: int = 4):
    """
    Carica l'indice FAISS salvato in 'data/faiss_index' 
    e recupera i k chunk di testo più simili alla query.
    """
    
    # 1. Definizione del Percorso (Corretta per tools/validation/)
    
    # Risale di due livelli: tools/validation/ -> tools/ -> classyfarm-rag/ (ROOT)
    base_dir = Path(__file__).resolve().parent.parent.parent
    
    # Costruisce il percorso finale: [ROOT]/data/faiss_index
    index_path = base_dir / "data" / "faiss_index" 

    # 2. Controllo e Caricamento
    if not index_path.exists():
        raise FileNotFoundError(
            f"ERRORE: La cartella dell'indice non esiste in: {index_path.resolve()}\n"
            "Assicurati che l'indice sia stato salvato correttamente lì (python -m tools.orchestrator full)."
        )

    print(f"Caricamento dell'indice dal percorso corretto: {index_path.resolve()}...")
    
    # Inizializza gli embeddings (richiede la chiave API caricata sopra)
    embeddings = OpenAIEmbeddings() 

    # Caricamento effettivo
    try:
        vectorstore = FAISS.load_local(
            folder_path=str(index_path),
            embeddings=embeddings,
            allow_dangerous_deserialization=True 
        )
        
        print(f"Indice caricato. Ricerca di {k} chunk per la query: '{query}'")

        # 3. Ricerca e Ritorno dei Chunk
        retrieved_docs = vectorstore.similarity_search(query, k=k)
        
        return retrieved_docs

    except Exception as e:
        print(f"Errore durante il caricamento o la ricerca: {e}")
        return []

# Esempio di utilizzo (quando esegui il file export_chunks.py)
if __name__ == "__main__":
    test_query = "Qual è la procedura che un Organismo di Certificazione deve seguire per registrarsi?"
    chunks = export_chunks(query=test_query, k=4) # Aumentato k a 4 per debug
    
    for i, chunk in enumerate(chunks):
        print(f"--- CHUNK {i+1} ---")
        print(f"Contenuto: {chunk.page_content[:200]}...")
        # 🚨 METADATI COMPLETI (Trova la chiave dell'ID)
        print(f"Metadati completi: {chunk.metadata}")
        print("-" * 20)