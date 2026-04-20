"""
Script di migrazione: JSONL -> SQLite
Legge l'ultimo dataset JSONL e lo inserisce nel database SQLite.
Popola anche l'indice FTS5 per la ricerca full-text.

Uso: python -m scripts.migrate_jsonl_to_sqlite
"""

import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from core.database import init_database, insert_chunks, get_chunk_count, save_dataset_version


def find_latest_dataset() -> Path:
    """Trova il dataset JSONL piu' recente."""
    datasets_dir = BASE_DIR / "data" / "datasets"
    versions = sorted(datasets_dir.glob("v*"), reverse=True)

    for v in versions:
        chunks_file = v / "chunks.jsonl"
        if chunks_file.exists():
            return chunks_file

    raise FileNotFoundError(f"Nessun chunks.jsonl trovato in {datasets_dir}")


def load_jsonl(path: Path) -> list:
    """Carica chunks da file JSONL."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"  ⚠️ Riga {line_num} non valida: {e}")
    return chunks


def main():
    print("=" * 60)
    print("MIGRAZIONE JSONL -> SQLite")
    print("=" * 60)

    # 1. Init DB
    print("\n1. Inizializzazione database...")
    init_database()

    # 2. Trova dataset
    print("\n2. Ricerca ultimo dataset...")
    chunks_path = find_latest_dataset()
    version_tag = chunks_path.parent.name
    print(f"   Dataset: {version_tag}")
    print(f"   Path: {chunks_path}")

    # 3. Carica JSONL
    print("\n3. Caricamento JSONL...")
    chunks = load_jsonl(chunks_path)
    print(f"   Chunks caricati: {len(chunks)}")

    if not chunks:
        print("❌ Nessun chunk trovato. Aborting.")
        return

    # 4. Inserisci nel DB
    print("\n4. Inserimento nel database SQLite + FTS5...")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        insert_chunks(batch)
        print(f"   Inseriti: {min(i + batch_size, len(chunks))}/{len(chunks)}")

    # 5. Verifica
    count = get_chunk_count()
    print(f"\n5. Verifica: {count} chunks nel database")

    # 6. Registra versione
    import hashlib
    content = open(chunks_path, "rb").read()
    fingerprint = hashlib.sha256(content).hexdigest()[:16]
    save_dataset_version(version_tag, fingerprint, note="Migrato da JSONL", chunk_count=count)

    print(f"\n✅ Migrazione completata!")
    print(f"   Database: {BASE_DIR / 'data' / 'palesai.db'}")
    print(f"   Chunks: {count}")
    print(f"   FTS5: attivo")
    print("=" * 60)


if __name__ == "__main__":
    main()
