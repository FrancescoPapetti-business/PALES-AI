from core.layers.layer8_application import RAGEngine

print("Inizializzazione engine...")
engine = RAGEngine()

retr = engine.main_retriever

print("\n=== METODI DISPONIBILI DEL RETRIEVER ===")
print([m for m in dir(retr) if not m.startswith("_")])

print("\n=== FLAG SUPPORTO ===")
print("has invoke:", hasattr(retr, "invoke"))
print("has get_relevant_documents:", hasattr(retr, "get_relevant_documents"))
print("has retrieve:", hasattr(retr, "retrieve"))
