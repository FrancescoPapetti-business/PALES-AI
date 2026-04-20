import os
import hashlib
from dataclasses import dataclass, field
from typing import List, Union
from pathlib import Path

# Calcola il percorso di default relativo alla posizione di questo file
# core/layers/layer1... -> (parents[2]) -> root -> data/pdf_input
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = BASE_DIR / "data" / "pdf_input"

@dataclass
class RawDocument:
    path: str
    rel_path: str
    filename: str
    sha256: str
    extension: str
    source_type: str = field(default="unknown")
    source_url: str = field(default="")      # URL originale (solo per sorgenti web)
    web_section: str = field(default="")     # Sezione del sito (es. "FAQ", "Guide")
    last_scraped: str = field(default="")    # Timestamp ISO 8601 dello scraping

def compute_sha256(path: Union[str, Path]) -> str:
    """Calcola l'hash SHA256 di un file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_sha256_from_text(text: str) -> str:
    """Calcola SHA256 da stringa di testo (per documenti web senza file fisico)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collect_documents(input_dir: Union[str, Path] = DEFAULT_INPUT_DIR) -> List[RawDocument]:
    """
    Scansiona la directory input_dir (e sottocartelle) cercando file .pdf/.docx/.doc.
    Restituisce una lista di oggetti RawDocument.
    """
    target_dir = Path(input_dir)
    docs: List[RawDocument] = []

    VALID_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".html",}
    
    if not target_dir.exists():
        print(f"⚠ Attenzione: Directory non trovata: {target_dir}")
        return []

    for root, _, files in os.walk(target_dir):
        for name in files:
            ext = Path(name).suffix.lower()
            if ext in VALID_EXTENSIONS:
                full_path = Path(root) / name

                # Calcola path relativo per pulizia nei log/DB
                try:
                    rel_path = full_path.relative_to(target_dir)
                except ValueError:
                    rel_path = name  # Fallback

                # Mappiamo il tipo di sorgente in modo coerente con il resto del pipeline
                lower_name = name.lower()
                if "faq" in lower_name:
                    doc_type = "faq"
                elif "vis" in lower_name or "scheda" in lower_name or "guida" in lower_name:
                    doc_type = "visual_guide"
                else:
                    doc_type = "text_document"

                docs.append(
                    RawDocument(
                        path=str(full_path),
                        rel_path=str(rel_path),
                        filename=name,
                        sha256=compute_sha256(full_path),
                        extension=ext,
                        source_type=doc_type,
                    )
                )

    return docs

def collect_web_documents(web_docs: List[RawDocument]) -> List[RawDocument]:
    """
    Validatore passthrough: accetta RawDocument già costruiti dal web scraper
    e verifica che abbiano i campi minimi necessari alla pipeline L2→L7.
    """
    valid = []
    for doc in web_docs:
        if not doc.sha256 or not doc.path:
            print(f"⚠ Web doc ignorato (mancano sha256/path): {doc.filename}")
            continue
        valid.append(doc)
    print(f"✅ Web docs validi per la pipeline: {len(valid)}/{len(web_docs)}")
    return valid

if __name__ == "__main__":
    # Test standalone
    print(f"🔍 Scansione directory: {DEFAULT_INPUT_DIR}")
    docs = collect_documents()
    print(f"✅ Trovati {len(docs)} documenti.")
    for d in docs[:5]:
        print(f" - {d.filename} [{d.source_type}] ({d.sha256[:10]}...)")
