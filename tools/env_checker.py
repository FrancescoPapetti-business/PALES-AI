import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # classyfarm-rag root
print(f"\n🔍 Scansione repository per uso di load_dotenv... (root: {ROOT})")

# Regex per intercettare load_dotenv / find_dotenv
pattern_env = re.compile(r"(load_dotenv|find_dotenv|dotenv_path)", re.IGNORECASE)

hits = []

for py_file in ROOT.rglob("*.py"):
    try:
        text = py_file.read_text(encoding="utf-8", errors="ignore")
    except:
        continue

    if pattern_env.search(text):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if pattern_env.search(line):
                hits.append((py_file, i + 1, line.strip()))

print("\n📌 RISULTATI:")
if not hits:
    print("✔ Nessun file usa load_dotenv o find_dotenv.")
else:
    for file, line_no, code in hits:
        print(f"\n--------------------------------------------------")
        print(f"📁 File: {file}")
        print(f"📍 Linea: {line_no}")
        print(f"🔧 Codice: {code}")

print("\n--------------------------------------------------")
print("⚠ Analisi automatica dei PATH problematici:")
for file, _, code in hits:
    if "utils" in code or "ENV_PATH" in code:
        print(f"❌ POSSIBILE ERRORE → {file}")
    elif "find_dotenv" in code:
        print(f"✔ OK (find_dotenv) → {file}")
    else:
        print(f"⚠ Da verificare manualmente → {file}")

print("\n🏁 Fine scansione.\n")
