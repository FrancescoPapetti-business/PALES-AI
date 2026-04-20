"""
web_scraper.py – Web Crawler per ClassyFarm RAG.
Scarica pagine e documenti dal sito classyfarm.it e li prepara
come RawDocument per la pipeline di ingestion esistente.

Uso standalone:
    python tools/web_scraper.py --url https://www.classyfarm.it --output data/web_cache

Uso come modulo:
    from tools.web_scraper import ClassyFarmScraper
    scraper = ClassyFarmScraper()
    raw_docs = scraper.scrape()
"""

import sys
import os
import re
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Set, Optional

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from core.layers.layer1_data_collection import RawDocument, compute_sha256_from_text

logger = logging.getLogger("web_scraper")

# ============================================================
#   CONFIGURAZIONE SCRAPER
# ============================================================

DEFAULT_BASE_URL = "https://www.classyfarm.it"
DEFAULT_WEB_CACHE_DIR = BASE_DIR / "data" / "web_cache"
PDF_DOWNLOAD_DIR = BASE_DIR / "data" / "pdf_input" / "web_downloaded"

# Headers HTTP per non essere bloccati
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ClassyFarmRAGBot/1.0; "
        "+https://github.com/frompersiwith69/CF-chatbot)"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/pdf",
}

# Delay tra richieste (rispetto al server)
REQUEST_DELAY_SECONDS = 1.5

# Profondità massima di crawling (1 = solo pagine linkate dalla homepage)
MAX_DEPTH = 2

# Estensioni da scaricare come documenti binari
DOWNLOADABLE_EXTENSIONS = {".pdf", ".docx", ".doc", ".xlsx", ".xls"}

# Pattern URL da ESCLUDERE (login, admin, social, ecc.)
EXCLUDE_PATTERNS = [
    r"/wp-admin",
    r"/wp-login",
    r"/feed/",
    r"\?s=",           # search queries
    r"#",              # ancore stessa pagina
    r"mailto:",
    r"javascript:",
    r"facebook\.com",
    r"twitter\.com",
    r"linkedin\.com",
    r"youtube\.com",
]

# Sezioni del sito da mappare (keyword → nome sezione)
SECTION_MAPPING = {
    "faq": "FAQ",
    "domande-frequenti": "FAQ",
    "guida": "Guide",
    "manuale": "Manuali",
    "documento": "Documentazione",
    "normativa": "Normativa",
    "news": "News",
    "aggiornamenti": "Aggiornamenti",
    "download": "Download",
    "moduli": "Moduli",
}


# ============================================================
#   CLASSE PRINCIPALE
# ============================================================

class ClassyFarmScraper:
    """
    Crawler per classyfarm.it.
    Produce oggetti RawDocument pronti per la pipeline L1→L7.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        web_cache_dir: Path = DEFAULT_WEB_CACHE_DIR,
        pdf_download_dir: Path = PDF_DOWNLOAD_DIR,
        max_depth: int = MAX_DEPTH,
        delay: float = REQUEST_DELAY_SECONDS,
        allowed_domain: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.web_cache_dir = Path(web_cache_dir)
        self.pdf_download_dir = Path(pdf_download_dir)
        self.max_depth = max_depth
        self.delay = delay
        self.allowed_domain = allowed_domain or urlparse(base_url).netloc

        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)

        self.visited_urls: Set[str] = set()
        self.failed_urls: List[str] = []

        # Crea cartelle di output
        self.web_cache_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_download_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🕷️ ClassyFarmScraper inizializzato")
        logger.info(f"   Base URL: {self.base_url}")
        logger.info(f"   Cache HTML: {self.web_cache_dir}")
        logger.info(f"   Download PDF: {self.pdf_download_dir}")

    # ----------------------------------------------------------
    #   ENTRY POINT
    # ----------------------------------------------------------

    def scrape(self) -> List[RawDocument]:
        """
        Esegue il crawling completo e restituisce tutti i RawDocument.
        Include sia pagine HTML che file PDF/DOCX scaricati.
        """
        print(f"\n🌐 Avvio crawling: {self.base_url}")
        all_raw_docs: List[RawDocument] = []

        # Crawl ricorsivo
        html_docs = self._crawl(self.base_url, depth=0)
        all_raw_docs.extend(html_docs)

        print(f"\n📊 Crawling completato:")
        print(f"   Pagine HTML indicizzate: {len([d for d in all_raw_docs if d.extension == '.html'])}")
        print(f"   File scaricati (PDF/DOCX): {len([d for d in all_raw_docs if d.extension != '.html'])}")
        print(f"   URL falliti: {len(self.failed_urls)}")
        if self.failed_urls:
            for u in self.failed_urls[:5]:
                print(f"   ❌ {u}")

        return all_raw_docs

    # ----------------------------------------------------------
    #   CRAWL RICORSIVO
    # ----------------------------------------------------------

    def _crawl(self, url: str, depth: int) -> List[RawDocument]:
        """Crawl ricorsivo con limite di profondità."""
        if depth > self.max_depth:
            return []
        if url in self.visited_urls:
            return []
        if self._should_exclude(url):
            return []

        self.visited_urls.add(url)
        results: List[RawDocument] = []

        print(f"   [depth={depth}] 🔍 {url}")
        time.sleep(self.delay)

        try:
            response = self.session.get(url, timeout=15, allow_redirects=True)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning(f"⚠ Errore HTTP per {url}: {e}")
            self.failed_urls.append(url)
            return []

        content_type = response.headers.get("Content-Type", "").lower()

        # ---- È un file scaricabile? (raggiunto direttamente, non tramite link) ----
        ext = Path(urlparse(url).path).suffix.lower()
        if ext in DOWNLOADABLE_EXTENSIONS or "application/pdf" in content_type:
            doc = self._save_binary_document(url, response)
            if doc:
                results.append(doc)
            return results

        # ---- Non è HTML → skip ----
        if "text/html" not in content_type:
            return []

        # ---- Pagina HTML: salva e segui link ----
        html_doc = self._save_html_page(url, response.text)
        if html_doc:
            results.append(html_doc)

        # Estrai sempre i link dalla pagina corrente
        links = self._extract_links(url, response.text)
        for link in links:
            link_ext = Path(urlparse(link).path).suffix.lower()

            if link_ext in DOWNLOADABLE_EXTENSIONS:
                # PDF/DOCX: scarica SEMPRE, indipendentemente dalla profondità
                if link not in self.visited_urls:
                    self.visited_urls.add(link)
                    print(f"   {'  ' * (depth+1)}📥 {Path(urlparse(link).path).name}")
                    try:
                        r = self.session.get(link, timeout=30, stream=True)
                        r.raise_for_status()
                        doc = self._save_binary_document(link, r)
                        if doc:
                            results.append(doc)
                    except requests.RequestException as e:
                        logger.warning(f"⚠ Errore download {link}: {e}")
                        self.failed_urls.append(link)
                    time.sleep(self.delay)

            elif depth < self.max_depth:
                # Pagine HTML: ricorsione solo se non al limite di profondità
                child_results = self._crawl(link, depth + 1)
                results.extend(child_results)

        return results
    # ----------------------------------------------------------
    #   SALVATAGGIO PAGINE HTML
    # ----------------------------------------------------------

    def _save_html_page(self, url: str, html_content: str) -> Optional[RawDocument]:
        """Salva una pagina HTML su disco e crea il corrispondente RawDocument."""
        # Genera nome file deterministico dall'URL
        url_slug = self._url_to_slug(url)
        filename = f"{url_slug}.html"
        file_path = self.web_cache_dir / filename

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
        except Exception as e:
            logger.error(f"❌ Impossibile salvare {file_path}: {e}")
            return None

        sha256 = compute_sha256_from_text(html_content)
        section = self._detect_section(url)
        timestamp = datetime.now(timezone.utc).isoformat()

        return RawDocument(
            path=str(file_path),
            rel_path=str(file_path.relative_to(BASE_DIR)),
            filename=filename,
            sha256=sha256,
            extension=".html",
            source_type="web_page",
            source_url=url,
            web_section=section,
            last_scraped=timestamp,
        )

    # ----------------------------------------------------------
    #   DOWNLOAD FILE BINARI (PDF, DOCX)
    # ----------------------------------------------------------

    def _save_binary_document(
        self, url: str, response: requests.Response
    ) -> Optional[RawDocument]:
        """Scarica e salva un file binario (PDF/DOCX) su disco."""
        ext = Path(urlparse(url).path).suffix.lower() or ".pdf"
        url_slug = self._url_to_slug(url)
        filename = f"{url_slug}{ext}"
        file_path = self.pdf_download_dir / filename

        # Non riscaricare se già presente e invariato
        if file_path.exists():
            print(f"   ⏭️ File già scaricato, skip: {filename}")
            from core.layers.layer1_data_collection import compute_sha256
            existing_sha = compute_sha256(file_path)
            section = self._detect_section(url)
            return RawDocument(
                path=str(file_path),
                rel_path=str(file_path.relative_to(BASE_DIR)),
                filename=filename,
                sha256=existing_sha,
                extension=ext,
                source_type="text_document",
                source_url=url,
                web_section=section,
                last_scraped=datetime.now(timezone.utc).isoformat(),
            )

        try:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"   ✅ Scaricato: {filename} ({len(response.content)//1024} KB)")
        except Exception as e:
            logger.error(f"❌ Errore salvataggio {filename}: {e}")
            return None

        from core.layers.layer1_data_collection import compute_sha256
        sha256 = compute_sha256(file_path)
        section = self._detect_section(url)

        return RawDocument(
            path=str(file_path),
            rel_path=str(file_path.relative_to(BASE_DIR)),
            filename=filename,
            sha256=sha256,
            extension=ext,
            source_type="text_document",
            source_url=url,
            web_section=section,
            last_scraped=datetime.now(timezone.utc).isoformat(),
        )

    # ----------------------------------------------------------
    #   ESTRAZIONE LINK
    # ----------------------------------------------------------

    def _extract_links(self, base_url: str, html_content: str) -> List[str]:
        """Estrae e normalizza tutti i link interni dalla pagina."""
        soup = BeautifulSoup(html_content, "lxml")
        links = []
        seen = set()

        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href:
                continue

            absolute = urljoin(base_url, href)
            # Rimuovi fragment
            parsed = urlparse(absolute)
            clean = urlunparse(parsed._replace(fragment=""))

            if clean in seen:
                continue
            if self._should_exclude(clean):
                continue
            if urlparse(clean).netloc != self.allowed_domain:
                continue

            seen.add(clean)
            links.append(clean)

        return links

    # ----------------------------------------------------------
    #   UTILITY
    # ----------------------------------------------------------

    def _should_exclude(self, url: str) -> bool:
        for pattern in EXCLUDE_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True
        return False

    def _url_to_slug(self, url: str) -> str:
        """Converte URL in nome file sicuro e leggibile."""
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        query = parsed.query[:30].replace("=", "-").replace("&", "_") if parsed.query else ""
        slug = f"{path}{'_' + query if query else ''}"
        slug = re.sub(r"[^\w\-]", "_", slug)
        slug = re.sub(r"_+", "_", slug).strip("_")
        # Tronca per evitare nomi file troppo lunghi
        if len(slug) > 100:
            slug = slug[:100] + "_" + hashlib.md5(url.encode()).hexdigest()[:6]
        return slug or hashlib.md5(url.encode()).hexdigest()[:16]

    def _detect_section(self, url: str) -> str:
        """Rileva la sezione del sito dall'URL."""
        url_lower = url.lower()
        for keyword, section_name in SECTION_MAPPING.items():
            if keyword in url_lower:
                return section_name
        return "Generale"


# ============================================================
#   CLI STANDALONE
# ============================================================

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="ClassyFarm Web Scraper")
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="URL base da scrapare")
    parser.add_argument("--depth", type=int, default=MAX_DEPTH, help="Profondità max crawling")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY_SECONDS, help="Delay tra richieste (sec)")
    parser.add_argument("--dry-run", action="store_true", help="Stampa URL senza scaricare")
    args = parser.parse_args()

    scraper = ClassyFarmScraper(
        base_url=args.url,
        max_depth=args.depth,
        delay=args.delay,
    )

    if args.dry_run:
        print("🔍 DRY RUN: nessun file verrà salvato.")
    else:
        docs = scraper.scrape()
        print(f"\n✅ Totale RawDocument prodotti: {len(docs)}")
        for d in docs[:10]:
            print(f"   - [{d.source_type}] {d.filename} | sezione: {d.web_section} | url: {d.source_url[:60]}")