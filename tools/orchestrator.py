"""
orchestrator.py – Pipeline end-to-end per ClassyFarm RAG.
Gestisce:
- ingestion documenti
- trasformazione (L2 + L3)
- riscrittura + chunking (L4)
- validazione chunk (L5)
- salvataggio dataset versionato (L6)
- indicizzazione FAISS/BM25 (L7, full + incremental)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
logger = logging.getLogger("orchestrator")

from dotenv import load_dotenv, find_dotenv

# Carico variabili ambiente (OPENAI_API_KEY ecc.)
load_dotenv(find_dotenv())

# Aggiungo root progetto al PYTHONPATH
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# ====== IMPORT LAYER ======

from core.layers.layer1_data_collection import collect_documents, RawDocument
from core.layers.layer2_quality_detection import detect_profile, DocumentProfile
from core.layers.layer3_cleaning_transformation import transform_document
from core.layers.layer4_semantic_rewriting import SemanticRewriter
from core.layers.layer5_validation_governance import validate_chunk_batch
from core.layers.layer6_dataset_manager import save_dataset_version
from core.layers.layer7_vector_store import (
    full_indexing_pipeline,
    incremental_indexing_update,
    get_indexed_sha256,   # implementata in L7
)

from tools.web_scraper import ClassyFarmScraper
from core.layers.layer1_data_collection import collect_web_documents

# ====== PATH UTILI (solo per log/consapevolezza) ======

DATA_DIR = BASE_DIR / "data"
PDF_INPUT_PATH = DATA_DIR / "pdf_input"


# ===========================================================
#   HELPER: costruzione chunk per un singolo documento
# ===========================================================

def build_chunks_for_document(raw_doc: RawDocument) -> List[Dict[str, Any]]:
    """
    Pipeline L1→L4 per UN singolo documento:
    - L2: profiling (tipo documento, strategia di processing)
    - L3: estrazione contenuto (markdown / immagini)
    - L4: riscrittura + chunking + metadata

    Ritorna: lista di chunk (dict) pronti per validazione (L5).
    """
    print(f"\n📄 Elaborazione documento: {raw_doc.rel_path}")

    # 1) Profilazione qualitativa (L2)
    profile: DocumentProfile = detect_profile(raw_doc)
    print(f"   → Tipo rilevato: {profile.doc_type} | strategy={profile.processing_strategy} | quality={profile.quality_score:.2f}")

    # 2) Trasformazione (L3)
    transformed = transform_document(profile)
    transformed["processing_strategy"] = profile.processing_strategy
    content_type = transformed.get("content_type")
    content = transformed.get("content")

    if content_type == "image_list":
        pass  # gestito più avanti

    elif content_type == "paged_text":
        if not content or not any(p.get("text", "").strip() for p in content):
            print("   ⚠ Nessun contenuto utile dopo la trasformazione (paged_text). Skip.")
            return []

    else:  # fallback stringa
        if not isinstance(content, str) or not content.strip():
            print("   ⚠ Nessun contenuto utile dopo la trasformazione (text). Skip.")
            return []


    # 3) Arricchisco con metadati di base e doc_type (servono a L4)
    base_metadata = {
        "filename": raw_doc.filename,
        "sha256": raw_doc.sha256,
        "parent_doc_id": raw_doc.sha256,
        "source_type": raw_doc.source_type,
        "extension": raw_doc.extension,
        "rel_path": raw_doc.rel_path,
        "source_url": getattr(raw_doc, "source_url", ""),
        "web_section": getattr(raw_doc, "web_section", ""),
        "last_scraped": getattr(raw_doc, "last_scraped", ""),
    }
    transformed["metadata"] = base_metadata
    transformed["doc_type"] = profile.doc_type

    # 4) Riscrittura + chunking semantico (L4)
    rewriter = SemanticRewriter()
    chunks = rewriter.process_document(transformed) or []


    print(f"   ✅ Generati {len(chunks)} chunk grezzi da L4.")
    return chunks


# ===========================================================
#   VALIDAZIONE CHUNK (L5)
# ===========================================================

def validate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Wrapper su L5.
    NB: Adatto a qualunque firma di validate_chunk_batch che ritorni
        o solo i chunk validi, o (validi, scartati).
    """
    if not chunks:
        return []

    print(f"   🔎 Invio {len(chunks)} chunk a L5 per validazione...")

    result = validate_chunk_batch(chunks)

    # Caso 1: validate_chunk_batch ritorna SOLO lista validi
    if isinstance(result, list):
        valid_chunks = result
        rejected = []
    # Caso 2: ritorna (validi, scartati)
    elif isinstance(result, tuple) and len(result) >= 1:
        valid_chunks = result[0]
        rejected = result[1] if len(result) > 1 else []
    else:
        print("   ⚠ Formato di ritorno inatteso da validate_chunk_batch, uso input originale.")
        valid_chunks = chunks
        rejected = []

    print(f"   ✅ Chunk validi: {len(valid_chunks)} | Scartati: {len(rejected)}")
    return valid_chunks


# ===========================================================
#   FULL PIPELINE: tutti i documenti → nuovo dataset + full index
# ===========================================================

def run_full_pipeline(note: str = "Full rebuild dataset + index") -> None:
    """
    Esegue l'intera pipeline su TUTTI i documenti in pdf_input:
    - L1-L4: costruzione chunk per ogni documento
    - L5: validazione chunk
    - L6: salvataggio dataset versionato (Golden Dataset)
    - L7: ricostruzione completa FAISS + BM25
    """
    print(f"\n🚀 Avvio FULL PIPELINE ClassyFarm RAG")
    print(f"   Cartella input: {PDF_INPUT_PATH}")

    all_docs = collect_documents(str(PDF_INPUT_PATH))
    print(f"   🧾 Documenti trovati: {len(all_docs)}")

    all_valid_chunks: List[Dict[str, Any]] = []

    for raw_doc in all_docs:
        try:
            # L1–L4
            chunks = build_chunks_for_document(raw_doc)
            if not chunks:
                continue

            # L5
            valid_chunks = validate_chunks(chunks)
            if not valid_chunks:
                print("   ⚠ Nessun chunk valido per questo documento, skip.")
                continue

            all_valid_chunks.extend(valid_chunks)

        except Exception as e:
            print(f"❌ Errore durante il processing di {raw_doc.filename}: {e}")
            continue

    if not all_valid_chunks:
        print("\n❌ Nessun chunk valido generato. Dataset NON creato, indicizzazione saltata.")
        return

    # 1) Salva nuova versione dataset (L6)
    dataset_path = save_dataset_version(all_valid_chunks, note=note)
    print(f"\n📦 Golden Dataset salvato in: {dataset_path}")

    # 2) Ricostruisci intero indice vettoriale (L7)
    print("\n🧠 Ricostruzione completa indice vettoriale + BM25...")
    full_indexing_pipeline()

    print("\n🏁 FULL PIPELINE completata con successo.")

    # ===========================================================
#   INCREMENTAL PIPELINE: solo PDF non ancora indicizzati
# ===========================================================

def run_incremental_pipeline() -> None:
    print(f"\n🚀 Avvio INCREMENTAL PIPELINE ClassyFarm RAG")
    print(f"   Cartella input: {PDF_INPUT_PATH}")

    all_docs = collect_documents(str(PDF_INPUT_PATH))
    print(f"   🧾 Documenti trovati: {len(all_docs)}")

    indexed_sha = set(get_indexed_sha256())
    print(f"   🧠 Documenti già indicizzati: {len(indexed_sha)}")

    processed = 0

    for raw_doc in all_docs:
        if raw_doc.sha256 in indexed_sha:
            print(f"⏭️ Skip {raw_doc.filename} (già indicizzato)")
            continue

        print(f"➕ Processing {raw_doc.filename}")
        process_single_document(raw_doc)
        processed += 1

    if processed == 0:
        print("\nℹ️ Nessun nuovo documento da processare.")
    else:
        print(f"\n✅ Incremental completato. Nuovi documenti processati: {processed}")

def run_web_pipeline(
    base_url: str = "https://www.classyfarm.it",
    depth: int = 2,
    incremental: bool = True,
    note: str = "Web ingestion ClassyFarm"
) -> None:
    """Pipeline completa da sorgente web ClassyFarm."""
    print(f"\n🌐 Avvio WEB PIPELINE ClassyFarm RAG")
    print(f"   URL: {base_url} | depth={depth} | incremental={incremental}")

    # Step 1: Crawling
    scraper = ClassyFarmScraper(base_url=base_url, max_depth=depth)
    scraped_docs = scraper.scrape()
    web_docs = collect_web_documents(scraped_docs)

    if not web_docs:
        print("❌ Nessun documento web valido. Pipeline interrotta.")
        return

    # Step 2: Filtro incrementale
    if incremental:
        indexed_sha = set(get_indexed_sha256())
        new_docs = [d for d in web_docs if d.sha256 not in indexed_sha]
        print(f"   🔎 Nuovi da indicizzare: {len(new_docs)}/{len(web_docs)}")
    else:
        new_docs = web_docs

    if not new_docs:
        print("\nℹ️ Nessun nuovo documento web da processare.")
        return

    # Step 3: Pipeline L2→L5
    all_valid_chunks = []
    for raw_doc in new_docs:
        try:
            chunks = build_chunks_for_document(raw_doc)
            if not chunks:
                continue
            valid = validate_chunks(chunks)
            if valid:
                all_valid_chunks.extend(valid)
        except Exception as e:
            print(f"❌ Errore {raw_doc.filename}: {e}")
            continue

    if not all_valid_chunks:
        print("\n❌ Nessun chunk valido prodotto. Dataset non aggiornato.")
        return

    # Step 4: Salva dataset + aggiorna indice incrementalmente
    save_dataset_version(all_valid_chunks, note=note)
    incremental_indexing_update(all_valid_chunks)
    print(f"\n🏁 WEB PIPELINE completata. Chunk aggiunti: {len(all_valid_chunks)}")


# ===========================================================
#   SINGLE DOC PIPELINE: rielabora 1 documento + indicizzazione incrementale
# ===========================================================

def process_single_document(raw_doc: RawDocument) -> None:
    """
    Esegue la pipeline su UN solo documento:
    - L1 (già fatto a monte): hai RawDocument
    - L2-L4: chunk + riscrittura
    - L5: validazione
    - L7: indicizzazione incrementale SOLO dei nuovi chunk
    (non tocca il dataset versionato su disco — quello rimane gestito da run_full_pipeline)
    """
    print(f"\n🔁 Rielaborazione singolo documento: {raw_doc.filename}")

    chunks = build_chunks_for_document(raw_doc)
    if not chunks:
        print("❌ Nessun chunk da indicizzare per questo documento.")
        return

    valid_chunks = validate_chunks(chunks)
    if not valid_chunks:
        print("❌ Nessun chunk valido dopo validazione. Nessun update indice eseguito.")
        return

    # Indicizzazione incrementale (L7)
    print("\n🧠 Aggiornamento incrementale indice vettoriale con i nuovi chunk...")
    incremental_indexing_update(valid_chunks)
    print("✅ Indicizzazione incrementale completata.")


def reprocess_single_pdf(pdf_name: str) -> None:
    """
    Trova un PDF nella cartella pdf_input e lancia process_single_document.
    """
    print(f"\n🔎 Ricerca PDF '{pdf_name}' in {PDF_INPUT_PATH}...")
    all_docs = collect_documents(str(PDF_INPUT_PATH))
    match = next((d for d in all_docs if d.filename == pdf_name), None)

    if not match:
        print(f"❌ Errore: il PDF '{pdf_name}' non esiste in {PDF_INPUT_PATH}.")
        return

    process_single_document(match)
    print("\n🎉 Operazione completata.")


# ===========================================================
#   ENTRYPOINT CLI SEMPLICE
# ===========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Orchestrator ClassyFarm RAG")
    parser.add_argument(
        "mode",
        choices=["full", "single", "incremental", "rebuild-index", "web", "web-incremental"],
        help="full = ricostruisce dataset + indice; single = rielabora un solo PDF"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Nome del PDF da rielaborare (richiesto se mode=single)"
    )
    parser.add_argument(
        "--note",
        type=str,
        default="Full rebuild dataset + index",
        help="Nota da salvare nel manifest del dataset (solo per mode=full)."
    )
    parser.add_argument("--url", type=str, default="https://www.classyfarm.it",
                        help="URL base per scraping web")
    parser.add_argument("--depth", type=int, default=2,
                        help="Profondità crawling (default: 2)")
    args = parser.parse_args()

    if args.mode == "full":
        run_full_pipeline(note=args.note)

    elif args.mode == "single":
        if not args.pdf:
            print("❌ Devi specificare --pdf NOMEFILE.pdf quando usi mode=single.")
        else:
            reprocess_single_pdf(args.pdf)

    elif args.mode == "incremental":
        run_incremental_pipeline()
        
    elif args.mode == "rebuild-index":
        print("\n🔧 Forzatura ricostruzione indici (Layer 7) da ultimo dataset...")
        full_indexing_pipeline()

    elif args.mode == "web":
        run_web_pipeline(base_url=args.url, depth=args.depth,
                         incremental=False, note=args.note)

    elif args.mode == "web-incremental":
        run_web_pipeline(base_url=args.url, depth=args.depth,
                         incremental=True, note=args.note)