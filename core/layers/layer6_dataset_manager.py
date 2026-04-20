"""
layer6_dataset_manager.py
Gestisce il salvataggio versionato dei chunk validati.
Crea snapshot immutabili (Golden Datasets) pronti per il Vector Store.
"""

import sys
import json
import shutil
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Gestione path
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Directory principale dove verranno salvate le versioni
DATASETS_DIR = BASE_DIR / "data" / "datasets"

def get_next_version_tag() -> str:
    """Calcola il prossimo tag di versione (es. v1, v2) basandosi sulle cartelle esistenti."""
    if not DATASETS_DIR.exists():
        return "v1"
    
    existing_versions = [d.name for d in DATASETS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not existing_versions:
        return "v1"
    
    # Estrae i numeri di versione (v1 -> 1, v2 -> 2)
    try:
        versions = [int(v.split("_")[0].replace("v", "")) for v in existing_versions]
        next_ver = max(versions) + 1
        return f"v{next_ver}"
    except ValueError:
        return f"v{len(existing_versions) + 1}"
    
def _normalize_chunk_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    meta = chunk.get("metadata", {})

    return {
        **chunk,
        "metadata": {
            **meta,
            "page_number": meta.get("page_number"),  # può essere None, ma ESISTE
            "filename": meta.get("filename"),
            "source_type": meta.get("source_type", "unknown")
        }
    }

def _compute_file_sha256(file_path: Path) -> str:
    """Calcola SHA256 del file per fingerprinting."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_dataset_version(chunks: List[Dict], note: str = "") -> str:
    """
    Salva una lista di chunk come una nuova versione immutabile del dataset.
    
    Args:
        chunks: Lista di dizionari (output del Layer 5)
        note: Nota opzionale per descrivere cosa cambia in questa versione (es. "Nuovo prompt Vision")
    
    Returns:
        Path della cartella creata.
    """
    if not chunks:
        print("⚠ Nessun chunk da salvare. Dataset non creato.")
        return ""

    # 1. Determina Versione
    version_tag = get_next_version_tag()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_dir_name = f"{version_tag}_{timestamp}"
    
    save_path = DATASETS_DIR / version_dir_name
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"📦 Creazione Snapshot Dataset: {version_tag} ({len(chunks)} chunks)...")

    # 2. Salvataggio Chunks (JSONL) - Formato ottimizzato per Ingestion
    chunks_file = save_path / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            normalized = _normalize_chunk_metadata(chunk)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    # Calcolo fingerprint del file appena creato
    file_hash = _compute_file_sha256(chunks_file)

    # 3. Salvataggio Manifest (Metadati della versione)
    manifest = {
        "version": version_tag,
        "created_at": timestamp,
        "note": note,
        "total_chunks": len(chunks),
        "chunks_file_sha256": file_hash,
        "schema_version": "1.1",
        "stats": {
            "visual_guides": len([c for c in chunks if c["metadata"].get("source_type") == "visual_guide"]),
            "text_docs": len([c for c in chunks if c["metadata"].get("source_type") == "text_document"])
        }
    }
    
    with open(save_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 4. Creazione link simbolico 'latest' (Opzionale, comodo per debug)
    latest_link = DATASETS_DIR / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        try:
            if latest_link.is_dir() and not latest_link.is_symlink():
                 shutil.rmtree(latest_link) # Se per errore è una dir reale
            else:
                 latest_link.unlink() # Rimuovi symlink precedente
        except Exception:
            pass # Ignora errori su Windows se symlink fallisce
            
    # Su Windows i symlink richiedono privilegi admin, quindi copiamo o ignoriamo
    if os.name != 'nt':
        try:
            latest_link.symlink_to(save_path)
        except OSError:
            pass 

    print(f"✅ Dataset salvato in: {save_path}")
    return str(save_path)

def list_datasets() -> List[Dict]:
    """Elenca tutti i dataset disponibili."""
    if not DATASETS_DIR.exists():
        return []
    
    datasets = []
    for d in sorted(DATASETS_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "manifest.json").exists():
            with open(d / "manifest.json", "r") as f:
                meta = json.load(f)
                datasets.append(meta)
    return datasets
def load_latest_manifest(version_path: str) -> Dict[str, Any]:
    """
    Carica il manifest.json di una specifica versione del dataset.
    """
    version_path = Path(version_path)
    manifest_file = version_path / "manifest.json"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest non trovato: {manifest_file}")

    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)

def get_dataset_fingerprint() -> str:
    """
    Restituisce una stringa identificativa univoca dell'ultimo dataset (Latest).
    Usata da Layer 7 per invalidare gli indici se il dataset cambia.
    """
    try:
        # Cerca la versione 'latest'
        if not DATASETS_DIR.exists():
            return "NO_DATASET"
        
        # Logica semplice: prendiamo l'ultima cartella vX
        versions = [d for d in DATASETS_DIR.iterdir() if d.is_dir() and d.name.startswith("v")]
        if not versions:
            return "NO_DATASET"
        
        latest_version = sorted(versions, key=lambda x: x.name, reverse=True)[0]
        manifest = load_latest_manifest(str(latest_version))
        
        # Combina versione e hash del contenuto
        return f"{manifest.get('version', 'v0')}:{manifest.get('chunks_file_sha256', 'nohash')}"
    except Exception as e:
        print(f"⚠️ Errore calcolo fingerprint dataset: {e}")
        return "ERROR_FINGERPRINT"
    
if __name__ == "__main__":
    # Test Standalone
    print("🗄️ Testing Layer 6 (Versioning)...")
    
    # Mock Data
    mock_chunks = [
        {"chunk_id": "1", "text": "Test A", "metadata": {"source_type": "text_document"}},
        {"chunk_id": "2", "text": "Test B", "metadata": {"source_type": "visual_guide"}}
    ]
    
    # 1. Salva v1
    path_v1 = save_dataset_version(mock_chunks, note="Test iniziale")
    
    # 2. Salva v2 (simuliamo modifica)
    mock_chunks.append({"chunk_id": "3", "text": "Test C", "metadata": {"source_type": "text_document"}})
    path_v2 = save_dataset_version(mock_chunks, note="Aggiunto chunk C")
    
    # 3. Lista
    print("\n📜 Dataset Disponibili:")
    for ds in list_datasets():    
        print(f" - {ds['version']} ({ds['created_at']}): {ds['note']} [Chunks: {ds['total_chunks']}]")