#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
extract_links.py – Utility per l'estrazione e la mappatura dei link.

Posizione: tools/utilities/extract_links.py
Funzioni:
- Scansiona data/data_rag_ready/ per file TXT
- Scansiona (opzionalmente) data/pdf_input/ per file PDF
- Estrae URL, li classifica e genera una "Linkmap"
- Produce output JSON (per metadati) e TXT (per ingestion RAG)
"""

import re
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

# Gestione dipendenza opzionale per PDF
try:
    from PyPDF2 import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# =========================
# CONFIGURAZIONE PATH
# =========================

# Risolve la root 'classyfarm-rag'
# tools/utilities/extract_links.py -> parents[0]=utilities -> parents[1]=tools -> parents[2]=root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Percorsi di default basati sulla nuova struttura
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_rag_ready"
DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "pdf_input"
DEFAULT_JSON_OUT = PROJECT_ROOT / "data" / "linkmap" / "dataset_linkmap.json"
# Il TXT va in rag_ready per essere indicizzato dal RAG
DEFAULT_TXT_OUT = PROJECT_ROOT / "data" / "data_rag_ready" / "dataset_linkmap.txt"


# =========================
# REGEX & COSTANTI
# =========================

URL_REGEX = re.compile(
    r"""(?i)\b((?:https?://|www\.)[^\s<>"'()]+)"""
)

MAX_CONTEXTS_PER_LINK = 5


# =========================
# FUNZIONI DI SUPPORTO
# =========================

def normalize_url(url: str) -> str:
    url = url.strip()
    # Rimuove punto finale se presente (comune fine frase)
    if url.endswith("."):
        url = url[:-1]
    if url.lower().startswith("www."):
        url = "https://" + url
    return url


def extract_urls_from_text(text: str):
    return [normalize_url(m.group(1)) for m in URL_REGEX.finditer(text)]


def classify_link(url: str, context: str) -> str:
    """Classifica il link in base al dominio e al contesto semantico."""
    ctx = context.lower()
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower()
    path = (parsed.path or "").lower()

    # Regole ClassyFarm
    if "classyfarm" in domain or "classyfarm" in path:
        if "manual" in path or "manuale" in ctx or "istruzioni" in ctx:
            return "manuale_classyfarm"
        if "guida" in path or "guida" in ctx:
            return "guida_classyfarm"
        if "dashboard" in path or "cruscotto" in ctx:
            return "cruscotto_classyfarm"
        if "delega" in path or "deleghe" in ctx:
            return "deleghe_classyfarm"
        return "pagina_classyfarm"

    # Ministero / PA
    if "salute.gov.it" in domain:
        if "guida" in ctx or "manuale" in ctx:
            return "guida_ministeriale"
        return "pagina_ministeriale"

    # Regole generiche basate sul contesto
    if any(k in ctx for k in ["manuale", "manual", "istruzioni"]):
        return "manuale_generico"
    if any(k in ctx for k in ["guida", "help"]):
        return "guida_generica"
    if any(k in ctx for k in ["cruscotto", "dashboard"]):
        return "cruscotto_generico"
    if any(k in ctx for k in ["delega", "deleghe"]):
        return "deleghe_generiche"
    if any(k in ctx for k in ["modulo", "modulistica"]):
        return "modulo_documentazione"

    return "link_generico"


def get_line_context(lines, idx, window=1):
    """Estrae le righe attorno al link per dare contesto all'LLM."""
    start = max(0, idx - window)
    end = min(len(lines), idx + window + 1)
    snippet = " ".join(line.strip() for line in lines[start:end])
    return snippet[:500]


# =========================
# LOGICA DI SCANSIONE
# =========================

def scan_txt_files(data_dir: Path):
    link_map = {}
    
    if not data_dir.exists():
        print(f"[WARN] Directory TXT non trovata: {data_dir}")
        return link_map

    # Esclude il file di output stesso per evitare loop
    txt_files = [f for f in sorted(data_dir.rglob("*.txt")) if f.name != DEFAULT_TXT_OUT.name]

    for txt_path in txt_files:
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Impossibile leggere {txt_path}: {e}")
            continue

        lines = text.splitlines()

        for i, line in enumerate(lines):
            urls = extract_urls_from_text(line)
            if not urls:
                continue

            context = get_line_context(lines, i)

            for url in urls:
                _add_to_map(link_map, url, context, txt_path, PROJECT_ROOT)

    return link_map


def scan_pdfs(pdf_dir: Path, link_map: dict):
    if not HAS_PDF:
        print("[WARN] PyPDF2 non installato: saltata scansione PDF.")
        return

    if not pdf_dir.exists():
        print(f"[WARN] Directory PDF non trovata: {pdf_dir}")
        return

    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"[INFO] Nessun PDF trovato in {pdf_dir}")
        return

    print(f"[INFO] Scansione {len(pdf_files)} PDF in {pdf_dir}...")

    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            full_text = []
            for page in reader.pages:
                try:
                    t = page.extract_text() or ""
                    full_text.append(t)
                except:
                    pass
            text = "\n".join(full_text)
        except Exception as e:
            print(f"[WARN] Errore lettura PDF {pdf_path.name}: {e}")
            continue

        lines = text.splitlines()

        for i, line in enumerate(lines):
            urls = extract_urls_from_text(line)
            if not urls:
                continue

            context = get_line_context(lines, i)
            for url in urls:
                _add_to_map(link_map, url, context, pdf_path, PROJECT_ROOT)


def _add_to_map(link_map, url, context, file_path, root_path):
    """Helper per aggiungere o aggiornare un link nella mappa."""
    parsed = urlparse(url)
    
    # Calcola path relativo per leggibilità
    try:
        rel_path = str(file_path.relative_to(root_path))
    except ValueError:
        rel_path = file_path.name

    if url not in link_map:
        link_map[url] = {
            "url": url,
            "domain": parsed.netloc.lower(),
            "category": classify_link(url, context),
            "source_files": set(),
            "contexts": [],
            "frequency": 0,
        }

    info = link_map[url]
    info["frequency"] += 1
    info["source_files"].add(rel_path)

    if len(info["contexts"]) < MAX_CONTEXTS_PER_LINK:
        info["contexts"].append(context)


# =========================
# OUTPUT
# =========================

def write_json(link_map, output_path: Path):
    # Prepara i dati (converte set in list)
    export_links = []
    for info in link_map.values():
        item = info.copy()
        item["source_files"] = sorted(list(item["source_files"]))
        export_links.append(item)

    data = {
        "project": "ClassyFarm RAG",
        "generated_at": datetime.now().isoformat(),
        "total_unique_links": len(link_map),
        "links": export_links,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON Linkmap scritto in: {output_path}")


def write_txt(link_map, output_path: Path):
    lines = []
    lines.append("MAPPA DEI LINK UTILI - CLASSYFARM RAG")
    lines.append("Questo documento contiene un indice dei link estratti dalla documentazione ufficiale.\n")

    # Ordina per frequenza decrescente
    sorted_links = sorted(link_map.values(), key=lambda x: (-x["frequency"], x["url"]))

    for info in sorted_links:
        lines.append(f"🔗 URL: {info['url']}")
        lines.append(f"   Categoria: {info['category']}")
        lines.append(f"   Frequenza: {info['frequency']}")
        
        # Converti source_files in lista se è ancora un set
        sources = sorted(list(info["source_files"]))
        if sources:
            lines.append("   Fonti:")
            for sf in sources[:3]: # Limitiamo a 3 fonti per non intasare il TXT
                lines.append(f"     - {sf}")
            if len(sources) > 3:
                lines.append(f"     - ...altri {len(sources)-3} file")

        if info["contexts"]:
            lines.append("   Contesto:")
            for c in info["contexts"][:1]: # Solo 1 contesto per brevità nel RAG
                lines.append(f"     \"{c.strip()}\"")

        lines.append("-" * 40)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] TXT Linkmap scritto in: {output_path}")


def json_to_txt(json_path: Path, output_path: Path):
    """Converte un JSON esistente in TXT (utile se si vuole rigenerare solo il txt)."""
    if not json_path.exists():
        print(f"[ERR] File JSON non trovato: {json_path}")
        return

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    links = data.get("links", [])
    # Ricostruisce la mappa (i source_files sono già liste nel JSON)
    link_map = {}
    for item in links:
        item["source_files"] = set(item["source_files"])
        link_map[item["url"]] = item

    write_txt(link_map, output_path)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser(description="Tool di estrazione Link per ClassyFarm RAG")

    parser.add_argument("--data-dir", type=str, help="Cartella contenente i file TXT puliti")
    parser.add_argument("--pdf-dir", type=str, help="Cartella contenente i PDF originali")
    parser.add_argument("--json-path", type=str, help="Percorso output JSON")
    parser.add_argument("--txt-path", type=str, help="Percorso output TXT")
    parser.add_argument("--from-json", type=str, help="Se specificato, converte questo JSON in TXT senza scansionare")

    args = parser.parse_args()

    # Definizione Percorsi (Argomenti > Default)
    data_dir = Path(args.data_dir) if args.data_dir else DEFAULT_DATA_DIR
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else DEFAULT_PDF_DIR
    json_path = Path(args.json_path) if args.json_path else DEFAULT_JSON_OUT
    txt_path = Path(args.txt_path) if args.txt_path else DEFAULT_TXT_OUT

    # Modalità solo conversione
    if args.from_json:
        print(f"[MODE] Conversione JSON -> TXT")
        json_to_txt(Path(args.from_json), txt_path)
        return

    # 1. Scansione TXT
    print(f"[INFO] Scansione TXT in: {data_dir}")
    link_map = scan_txt_files(data_dir)

    # 2. Scansione PDF (se richiesto/disponibile)
    if args.pdf_dir or pdf_dir.exists():
        # Usa l'argomento esplicito o il default se esiste
        target_pdf = Path(args.pdf_dir) if args.pdf_dir else pdf_dir
        scan_pdfs(target_pdf, link_map)

    print(f"[INFO] Totale link unici trovati: {len(link_map)}")

    # 3. Scrittura Output
    write_json(link_map, json_path)
    write_txt(link_map, txt_path)


if __name__ == "__main__":
    main()

