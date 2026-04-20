import sys
import re
import fitz  # PyMuPDF
import pymupdf4llm  # <--- NUOVO: Converte PDF in Markdown (Post #1)
from docx import Document as DocxDocument
from pathlib import Path
from typing import Dict, Any
try:
    import html2text
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.layers.layer2_quality_detection import DocumentProfile

# --- EXTRACTORS ---
def extract_text_file(path: str) -> str:
    """Legge file di testo puro (.txt, .md, .html)."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception as e:
        print(f"❌ Errore lettura file testo {path}: {e}")
        return ""

def _clean_web_text(text: str) -> str:
    """Pulizia testo estratto da web: rimuove artefatti comuni."""
    if not text:
        return ""
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [l for l in text.split("\n") if len(l.strip()) > 2 or l.strip() == ""]
    return "\n".join(lines).strip()


def extract_web_page(path: str, source_url: str = "") -> str:
    """
    Estrae testo pulito e strutturato da un file HTML salvato da pagina web.
    Strategia a cascata:
      1. html2text → Markdown leggibile (preferito)
      2. BeautifulSoup → testo strutturato
      3. Raw fallback
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            raw_html = f.read()
    except Exception as e:
        print(f"❌ Errore lettura HTML {path}: {e}")
        return ""

    # Strategia 1: html2text
    if HAS_HTML2TEXT:
        try:
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.ignore_emphasis = False
            h.body_width = 0
            h.skip_internal_links = True
            text = _clean_web_text(h.handle(raw_html))
            if text and len(text) > 100:
                return text
        except Exception as e:
            print(f"⚠ html2text fallito per {path}: {e}")

    # Strategia 2: BeautifulSoup
    if HAS_BS4:
        try:
            soup = BeautifulSoup(raw_html, "lxml")
            for tag in soup(["script", "style", "nav", "header", "footer",
                             "aside", "noscript", "iframe", "form"]):
                tag.decompose()
            lines = []
            for el in soup.find_all(["h1","h2","h3","h4","p","li","td","th"]):
                t = el.get_text(separator=" ", strip=True)
                if not t:
                    continue
                if el.name == "h1":
                    lines.append(f"\n# {t}\n")
                elif el.name == "h2":
                    lines.append(f"\n## {t}\n")
                elif el.name in ("h3","h4"):
                    lines.append(f"\n### {t}\n")
                elif el.name == "li":
                    lines.append(f"- {t}")
                else:
                    lines.append(t)
            text = _clean_web_text("\n".join(lines))
            if text and len(text) > 100:
                return text
        except Exception as e:
            print(f"⚠ BeautifulSoup fallito per {path}: {e}")

    # Strategia 3: raw fallback
    print(f"⚠ Doppio fallback raw per {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return _clean_web_text(f.read())
    except Exception:
        return ""

def extract_docx_markdown(path: str) -> str:
    """Legge DOCX e tenta una conversione base in Markdown."""
    try:
        doc = DocxDocument(path)
        full_text = []
        for para in doc.paragraphs:
            # Semplice euristica per headers
            if para.style.name.startswith('Heading'):
                full_text.append(f"## {para.text}")
            else:
                full_text.append(para.text)
        return "\n\n".join(full_text)
    except Exception as e:
        print(f"❌ Errore DOCX su {path}: {e}")
        return ""

def extract_pdf_markdown_by_page(path: str) -> list:
    """
    Estrae il contenuto Markdown pagina per pagina.
    Ritorna una lista di dict:
    [
      {"page_number": 1, "text": "..."},
      {"page_number": 2, "text": "..."}
    ]
    """
    pages = []
    try:
        doc = fitz.open(str(path))
        for i, page in enumerate(doc):
            raw_text = page.get_text("text")
            if not raw_text:
                continue
            if isinstance(raw_text, list):
                raw_text = "\n".join(str(x) for x in raw_text)
            text = markdown_clean(raw_text)
            if isinstance(text, str) and text.strip():
                pages.append({
                    "page_number": i + 1,
                    "text": text
                })
        doc.close()
    except Exception as e:
        print(f"❌ Errore PDF page extraction: {e}")
    return pages


def extract_text_raw_fallback(path: str) -> str:
    """Fallback classico PyMuPDF se la conversione MD fallisce."""
    try:
        doc = fitz.open(str(path))
        texts = [page.get_text("text") for page in doc]
        doc.close()
        return "\n\n".join(texts)
    except Exception:
        return ""

def prepare_images_for_vision(path: str) -> list:
    """
    Per le guide visuali: NON estrae testo (sarebbe vuoto/inutile).
    Converte le pagine in percorsi immagini per il Layer 4 (Vision LLM).
    """
    doc = fitz.open(path)
    image_paths = []
    output_dir = Path(path).parent / "extracted_images"
    output_dir.mkdir(exist_ok=True)

    for i, page in enumerate(doc):
        # Renderizza la pagina come immagine (fondamentale per screenshot con frecce)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom per qualità OCR
        img_path = output_dir / f"{Path(path).stem}_page_{i}.png"
        pix.save(img_path)
        image_paths.append(str(img_path))
    
    doc.close()
    return image_paths

# --- CLEANING ---

def markdown_clean(text: str) -> str:
    """Pulizia specifica per Markdown."""
    if not text: return ""
    # Rimuove immagini markdown rotte o inutili per ora
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Rimuove link eccessivi
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    # Normalizza spazi
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# --- ORCHESTRATOR ---

def transform_document(profile: DocumentProfile) -> Dict[str, Any]:
    """
    Restituisce un dizionario contenente il contenuto processato.
    Format: {'content': str, 'type': str, 'metadata': dict}
    """
    path = profile.raw.path
    strategy = getattr(profile, 'processing_strategy', 'text_extract') # fallback se layer 2 vecchio
    
    result = {
        "content": None,
        "content_type": None,
        "image_paths": [],
        "strategy_used": strategy
    }


    print(f"⚙️ Applying strategy: {strategy} for {profile.raw.filename}")

    # 1. STRATEGIA: VISION LLM (Guide Visuali)
    if strategy == "vision_llm":
        # Qui NON estraiamo testo. Prepariamo le immagini per il Layer 4.
        # Il Layer 4 userà queste immagini per generare la descrizione.
        images = prepare_images_for_vision(path)
        result["image_paths"] = images
        result["content_type"] = "image_list"
        result["content"] = "[CONTENT_PENDING_VISION_ANALYSIS]" # Placeholder
    
    # STRATEGIA: HTML EXTRACT (pagine web)
    elif strategy == "html_extract":
        source_url = getattr(profile.raw, "source_url", "")
        raw = extract_web_page(path, source_url=source_url)
        result["content"] = raw
        result["content_type"] = "text"
        result["source_url"] = source_url
    # 2. STRATEGIA: TEXT / TABLE / DOCX
    else:
        ext = Path(path).suffix.lower()

        if ext == ".docx":
            raw = extract_docx_markdown(path)
            result["content"] = raw
            result["content_type"] = "text"

        elif ext in {".txt", ".md", ".html"}:
            raw = extract_text_file(path)
            result["content"] = raw
            result["content_type"] = "text"

        else:  # PDF
            pages = extract_pdf_markdown_by_page(path)
            result["content"] = pages
            result["content_type"] = "paged_text"

        
    # Fallback di sicurezza
    if result["content_type"] == "paged_text" and not result["content"]:
        result["content"] = ""
        result["content_type"] = "text"




    return result

if __name__ == "__main__":
    from core.layers.layer1_data_collection import collect_documents
    from core.layers.layer2_quality_detection import detect_profile
    
    print("🔍 Testing Layer 3 (Transformation)...")
    docs = collect_documents()
    
    for d in docs[:3]: # Test primi 3
        prof = detect_profile(d)
        res = transform_document(prof)
        
        if res["content_type"] == "image_list":
            print(f"📸 Visual Guide detected: {len(res['image_paths'])} pagine convertite in immagini.")
            print(f"   (Pronto per Layer 4 Vision LLM)")
        else:
            print(f"📝 Text Doc extracted: {len(res['content'])} chars (Markdown format).")
            print(f"   Preview: {res['content'][:100]}...")
        print("-" * 40)