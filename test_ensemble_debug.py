print("=== TEST ENSEMBLE RETRIEVER ===")

from core.layers.layer8_application import RAGEngine

engine = RAGEngine()

retr = engine.main_retriever

print("\nTipo retriever:", type(retr))

print("\nMetodi disponibili:")
for m in ["invoke", "retrieve", "get_relevant_documents", "search"]:
    print(f" - {m}: {hasattr(retr, m)}")

print("\nTest chiamata invoke('classyfarm')...\n")

try:
    docs = retr.invoke("classyfarm")
    print(f"OK: recuperati {len(docs)} documenti")
except Exception as e:
    print("Errore:", e)
