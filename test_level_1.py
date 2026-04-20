import os
from dotenv import load_dotenv # Aggiunta la libreria
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# --- PUNTO CRITICO: CARICAMENTO CHIAVE API ---
# Questo deve essere il primissimo codice ad essere eseguito.
load_dotenv() 

# Verifica immediata prima di inizializzare OpenAI
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("❌ ERRORE: La variabile OPENAI_API_KEY non è stata caricata. Controlla il file .env.")

# ---------------------------------------------

# Configurazione del percorso (La tua logica originale)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index")

print(f"Testing percorso: {INDEX_PATH}")

if not os.path.exists(INDEX_PATH):
    print("❌ ERRORE: La cartella non esiste!")
else:
    try:
        # L'embedding model ora trova la chiave nell'ambiente
        embeddings = OpenAIEmbeddings() 
        
        # Prova a caricare l'indice
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        
        print(f"✅ SUCCESSO: Indice caricato. Contiene {vectorstore.index.ntotal} vettori.")
    except Exception as e:
        print(f"❌ ERRORE DI CARICAMENTO FAISS: {e}")