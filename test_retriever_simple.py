# test_retriever_simple.py
from core.layers.layer8_application import RAGEngine

engine = RAGEngine()
retr = engine.main_retriever

print("Retriever type:", type(retr))

q = "come si accede a classyfarm"

if hasattr(retr, "invoke"):
    docs = retr.invoke(q)
elif hasattr(retr, "retrieve"):
    docs = retr.retrieve(q)
elif hasattr(retr, "get_relevant_documents"):
    docs = retr.get_relevant_documents(q)
else:
    raise RuntimeError("Retriever senza metodo valido")

print("Numero documenti:", len(docs))
if docs:
    print("Preview:", docs[0].page_content[:200])