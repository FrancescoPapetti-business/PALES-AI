import os
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pathlib import Path

# --- Setup per caricare la chiave .env ---
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env") 
# Se il tuo .env è nella root, usa: load_dotenv(BASE_DIR / ".env")


API_KEY = os.getenv("OPENAI_API_KEY")

def simple_api_check():
    """
    Tenta di eseguire una chiamata base (completamento) per verificare la chiave.
    """
    print("\n--- Esecuzione Test Connettività API OpenAI ---")
    
    if not API_KEY:
        print("❌ ERRORE: Chiave API non trovata nelle variabili d'ambiente.")
        return

    try:
        # 1. Inizializzazione Client
        client = OpenAI(api_key=API_KEY)
        
        # 2. Chiamata API minima (gpt-3.5-turbo è veloce ed economico)
        print("⏳ Inizio chiamata... (Model: gpt-3.5-turbo)")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Dimmi una parola in italiano."}],
            max_tokens=5
        )
        
        # 3. Successo
        risposta = response.choices[0].message.content.strip()
        print(f"\n✅ SUCCESS: La chiave funziona!")
        print(f"   Risposta del Modello: '{risposta}'")
        
        # Verifica se il testo è stato restituito (controllo aggiuntivo)
        if risposta:
            print("\n**Il tuo sistema è connesso e autenticato correttamente.**")
        
    except OpenAIError as e:
        # Cattura l'errore 401 e altri errori specifici dell'API
        if "Incorrect API key" in str(e):
            print("\n❌ ERRORE CRITICO (401): Chiave API Errata o Scaduta.")
            print("   Dettaglio: Devi sostituire il valore di OPENAI_API_KEY nel tuo file .env.")
        elif "Rate limit" in str(e):
            print("\n⚠️ ERRORE (429): Limite di Frequenza Raggiunto. Riprova tra poco.")
        else:
            print(f"\n❌ ERRORE API Sconosciuto: {e}")
            
    except Exception as e:
        print(f"\n❌ ERRORE GENERALE: {e}")

if __name__ == "__main__":
    simple_api_check()