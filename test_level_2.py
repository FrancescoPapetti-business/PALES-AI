import sys
import os
from dotenv import load_dotenv

# 1. Carica le variabili d'ambiente (CRUCIALE)
load_dotenv()

# 2. Aggiunge la root al path per trovare i moduli core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("--- TEST LIVELLO 2: Importazione Backend ---")

try:
    # Tenta di importare l'applicazione. 
    # Se il file layer8_application.py ha errori di sintassi o logica, fallirà qui.
    print("Tentativo di importazione di core.layers.layer8_application...")
    from core.layers.layer8_application import app
    
    print("✅ Oggetto 'app' (FastAPI) trovato e importato correttamente.")
    print("✅ Sintassi del file layer8_application.py corretta.")

except ImportError as e:
    print(f"❌ ERRORE DI IMPORT: {e}")
    print("Verifica che esistano i file __init__.py in 'core' e 'core/layers'.")
except Exception as e:
    print(f"❌ ERRORE DURANTE L'INIZIALIZZAZIONE: {e}")
    print("Probabile causa: Errore nel caricamento dell'indice o della chiave API dentro layer8_application.")