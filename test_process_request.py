from core.layers.layer8_application import process_request

query = "Come si accede a ClassyFarm?"

res = process_request(
    question=query,
    role="operatore",
    chat_history=[]
)

print("\n=== RISPOSTA ===")
print(res.get("answer"))

print("\n=== LATENZA ===")
print(res.get("latency"))

print("\n=== SAFETY FLAGS ===")
print(res.get("safety_flags"))

print("\n=== FONTI ===")
for s in res.get("sources", []):
    print(
        f"- {s.get('filename')} | pag. {s.get('page_number')} | type: {s.get('source_type')}"
    )
