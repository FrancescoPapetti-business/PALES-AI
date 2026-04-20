import sys
import os
import uuid
import json
import base64
import regex
import re
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage

LINK_PATTERNS = {
    # Modello robusto per URL: http(s)://, www., e domini semplici
    "url": r'https?://[^\s<>"]+|www\.[^\s<>"]+\.[a-zA-Z]{2,}',
    # Modello robusto per Email
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
}

def normalize_input_content(transformed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    content_type = transformed_data.get("content_type")
    content = transformed_data.get("content")

    if content_type == "image_list":
        return [{"text": None, "page_number": i+1} for i, _ in enumerate(transformed_data.get("image_paths", []))]

    if isinstance(content, list):
        return [
            {"text": p.get("text", ""), "page_number": p.get("page_number")}
            for p in content
            if isinstance(p, dict)
        ]

    if isinstance(content, str):
        return [{"text": content, "page_number": None}]

    return []


def extract_links_from_text(text: str) -> List[Dict[str, str]]:
    """Usa Regex per trovare link e email nel testo."""
    extracted_links = []
    
    # Estrazione URL
    for match in regex.finditer(LINK_PATTERNS["url"], text):
        extracted_links.append({
            "url": match.group(0),
            "type": "url",
            # Il campo description sarà popolato dall'LLM più avanti
            "description": "" 
        })
        
    # Estrazione Email
    for match in regex.finditer(LINK_PATTERNS["email"], text):
        extracted_links.append({
            "url": match.group(0),
            "type": "email",
            "description": "" 
        })
        
    return extracted_links

# Gestione path
# Path corretto alla ROOT del progetto
BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

from dotenv import load_dotenv
load_dotenv(ENV_PATH)

print(f"[DEBUG L4] Layer loaded .env from: {ENV_PATH}")


if not os.getenv("OPENAI_API_KEY"):
    print("⚠ ATTENZIONE: OPENAI_API_KEY non trovata in utils/.env")

# --- CONFIGURAZIONE LLM ---
# GPT-4o per la Vision (Costoso ma necessario per le frecce/numeri)
llm_vision = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1000, request_timeout=60)
# GPT-4o-mini per Text Rewriting e Metadata (Veloce ed economico)
llm_text = ChatOpenAI(model="gpt-4o-mini", temperature=0, request_timeout=30)

# --- PROMPT SPECIALIZZATI ---

# Prompt per Guide Visuali (Screenshot ClassyFarm)
VISION_PROMPT = """
Sei un analista tecnico di piattaforme ministeriali. Analizza questo screenshot.
1. Identifica il Titolo della pagina.
2. Se ci sono NUMERI o FRECCE (es. cerchi rossi con 1, 2, 3):
   - Collega ogni numero all'elemento indicato (es. "Il punto 1 indica il tasto Accedi").
   - Descrivi la procedura logica mostrata passo dopo passo.
3. Trascrivi il testo rilevante visibile nell'interfaccia.
Output: Una descrizione dettagliata e discorsiva della procedura.
"""

# Prompt per Metadati (Keywords, Summary)
METADATA_PROMPT = """
Analizza il seguente frammento di testo. Restituisci SOLO un oggetto JSON valido con questi campi:
{
  "summary": "Riassunto di una frase del contenuto",
  "keywords": ["tag1", "tag2", "tag3", "tag4"],
  "topic": "Categoria (es. Accesso, Normativa, Biosicurezza, Farmaco)"
}
Testo:
"""
# ======================================================
# SEMANTIC LIFTING — BUG-01
# ======================================================

MANDATORY_HINTS = [
    "inserire", "compilare", "obbligatorio", "necessario",
    "richiesto", "per completare", "è richiesto"
]

EXCLUSIVE_ACCESS_HINTS = [
    "esclusivamente", "solo", "soltanto",
    "assegnati", "non visualizza", "non ha accesso"
]

PROCEDURE_SERVICE_HINTS = [
    "procedura guidata", "assistenza", "supporto",
    "contattare", "numero verde"
]


def implies_mandatory(text: str) -> bool:
    return any(h in text.lower() for h in MANDATORY_HINTS) and "email" in text.lower()


def implies_exclusive_access(text: str) -> bool:
    return any(h in text.lower() for h in EXCLUSIVE_ACCESS_HINTS)


def implies_procedure_service(text: str) -> bool:
    return any(h in text.lower() for h in PROCEDURE_SERVICE_HINTS)


def lift_mandatory_rule(text: str) -> str:
    return (
        "È obbligatorio inserire un indirizzo email valido per completare la richiesta, "
        "poiché le comunicazioni e le conferme vengono inviate tramite email."
    )


def lift_exclusive_access_rule(text: str) -> str:
    return (
        "L’utente può visualizzare esclusivamente le risorse assegnate "
        "e non ha accesso ad altre informazioni non autorizzate."
    )


def lift_procedure_service_rule(text: str) -> str:
    return (
        "La procedura guidata fa riferimento a un servizio di assistenza o supporto "
        "che accompagna l’utente nelle fasi operative previste."
    )


def extract_and_lift_rules(
    text: str,
    original_chunk_id: str,
    base_meta: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Ritorna una LISTA di chunk:
    - regole esplicite se presenti
    - altrimenti il testo originale
    """

    lifted_chunks = []

    if implies_mandatory(text):
        lifted_chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text_content": lift_mandatory_rule(text),
            "metadata": {
                **base_meta,
                "semantic_role": "mandatory_rule",
                "explicit": True,
                "derived_from": original_chunk_id,
                "confidence": "high"
            }
        })

    if implies_exclusive_access(text):
        lifted_chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text_content": lift_exclusive_access_rule(text),
            "metadata": {
                **base_meta,
                "semantic_role": "access_rule",
                "explicit": True,
                "derived_from": original_chunk_id,
                "confidence": "high"
            }
        })

    if implies_procedure_service(text):
        lifted_chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "text_content": lift_procedure_service_rule(text),
            "metadata": {
                **base_meta,
                "semantic_role": "service_mapping",
                "explicit": True,
                "derived_from": original_chunk_id,
                "confidence": "medium"
            }
        })

    return lifted_chunks

# ======================================================
# CANONICAL CONCEPTS — BUG-02 (MVP hard-coded)
# ======================================================

CANONICAL_MAP = {
    "DASHBOARD": [
        "cruscotto",
        "dashboard",
        "cruscotto di sintesi",
        "cruscotto sqnba",
        "sqnba",
        "cruscotto verifica prerequisiti",
        "cruscotto prerequisiti",
    ],
    "ASSISTENZA_TELEFONICA": [
        "procedura guidata",
        "assistenza telefonica",
        "numero di assistenza",
        "call center",
        "supporto telefonico",
        "numero verde",
        "assistenza",
        "supporto",
    ],
    "INVIO_EMAIL": [
        "inviare la richiesta",
        "invio documentazione",
        "spedire la richiesta",
        "mandare la domanda",
        "inviare via email",
        "invio via email",
    ],
    "ACRONIMI_DOMAIN": {
        # Organizzazioni
        "OdC": ["organismo di certificazione", "organismo certificazione", "ente certificazione"],
        "BDN": ["banca dati nazionale", "base dati nazionale"],
        "ASL": ["azienda sanitaria locale", "servizio veterinario"],
        "IZS": ["istituto zooprofilattico"],
        
        # Documenti/Processi
        "CL": ["checklist", "check-list", "lista controllo", "scheda valutazione"],
        "FAQ": ["domande frequenti", "frequently asked questions"],
        "DDDAit": ["defined daily dose", "dose giornaliera definita"],
        
        # Sistema ClassyFarm
        "DPA": ["destinato produzione alimenti"],
        "SQNBA": ["sistema qualità nazionale benessere animale"],
        "PAC": ["politica agricola comune"],
        
        # UI/Technical
        "PEC": ["posta elettronica certificata"],
        "PIN": ["codice pin", "password"],
        "APP": ["applicazione", "applicazione mobile"],
    }
}

def _basic_normalize_text(text: str) -> str:
    """
    Normalizzazione leggera e deterministica:
    - lowercase/casefold
    - spazi multipli compressi
    Non distrugge il testo (niente stemming/lemmatizzazione).
    """
    if not isinstance(text, str):
        return ""
    t = text.casefold()
    t = regex.sub(r"\s+", " ", t).strip()
    return t

def normalize_concepts(text: str) -> Dict[str, Any]:
    """
    Ritorna:
      - normalized_text: testo normalizzato (lower + whitespace)
      - canonical_concepts: lista concetti canonici trovati
    Matching: substring su testo normalizzato.
    (MVP robusto, dominio-driven)
    """
    normalized = _basic_normalize_text(text)
    found = []

    for canonical, synonyms in CANONICAL_MAP.items():
        for s in synonyms:
            s_norm = _basic_normalize_text(s)
            if s_norm and s_norm in normalized:
                found.append(canonical)
                break

    # dedup preservando ordine
    canonical_concepts = list(dict.fromkeys(found))
    return {
        "normalized_text": normalized,
        "canonical_concepts": canonical_concepts
    }

def inject_canonical_tags(text: str, canonical_concepts: List[str]) -> str:
    """
    Opzione A (semplice): prefissa i tag canonici nel testo.
    Questo NON sostituisce le parole originali (zero rischio di rovinare contenuti).
    """
    if not canonical_concepts:
        return text
    prefix = " ".join([f"[{c}]" for c in canonical_concepts])
    return f"{prefix} {text}".strip()

def expand_acronym_query(query: str) -> str:
    """
    Espande una query che contiene acronimi con i loro sinonimi.
    Es: "Cosa significa OdC?" → "Cosa significa OdC organismo certificazione"
    
    Usare questo solo se infer_intent() classifica come "acronym_definition".
    """
    # Cerca acronimi noti (maiuscole 2-6 lettere)
    acronyms_in_query = re.findall(r'\b([A-Z]{2,6})\b', query)
    
    if not acronyms_in_query:
        return query
    
    # Carica mapping acronimi
    acronym_map = CANONICAL_MAP.get("ACRONIMI_DOMAIN", {})
    
    expansions = []
    for acronym in acronyms_in_query:
        synonyms = acronym_map.get(acronym, [])
        if synonyms:
            # Aggiungi primi 2 sinonimi più rilevanti
            expansions.extend(synonyms[:2])
    
    if expansions:
        return f"{query} {' '.join(expansions)}"
    
    return query

def expand_query_semantic(question: str) -> str:
    """
    Wrapper deterministico per espandere una query
    usando canonical concepts (BUG-02 / BUG-10).
    NON introduce nuova logica.
    """
    norm = normalize_concepts(question)
    canonical_concepts = norm.get("canonical_concepts", []) or []
    return inject_canonical_tags(question, canonical_concepts)


class SemanticRewriter:
    
    def _generate_chunk_id(self) -> str:
        """Genera un ID univoco per il chunk."""
        return str(uuid.uuid4())

    def normalize_query(self, user_query: str) -> Dict[str, Any]:
        """
        Usare in layer8: stessa identica normalizzazione dei chunk.
        """
        return normalize_concepts(user_query)

    def _encode_image(self, image_path: str) -> str:
        """Codifica immagine in base64 per l'API."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _analyze_image_vision(self, image_path: str) -> str:
        """Usa GPT-4o per descrivere lo screenshot."""
        try:
            base64_image = self._encode_image(image_path)
            message = HumanMessage(
                content=[
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            )
            response = llm_vision.invoke([message])
            return response.content
        except Exception as e:
            print(f"❌ Errore Vision su {image_path}: {e}")
            return ""

    def _rewrite_text_chunk(self, chunk, doc_type: str) -> str:
        print("DEBUG rewrite_text_chunk type:", type(chunk))

        if not isinstance(chunk, str):
            print("❌ chunk NON stringa:", chunk)
            return ""

        if len(chunk.strip()) < 50:
            return ""

        instruction = "Riscrivi in italiano professionale mantenendo i dati tecnici."
        if doc_type == "table":
            instruction = "Converti la tabella in elenco strutturato discorsivo."
        elif doc_type == "slide":
            instruction = "Unisci i punti elenco in un discorso fluido e completo."
        elif doc_type == "web_content":
            instruction = (
                "Riscrivi questo testo estratto da una pagina web in italiano professionale. "
                "Mantieni SEMPRE invariati: indirizzi email, URL, numeri di telefono, "
                "nomi propri, sigle tecniche e scadenze temporali. "
                "Elimina navigazione, breadcrumb e testi di menu non informativi."
            )

        prompt = f"{instruction}\n\nTESTO:\n\"\"\"{chunk}\"\"\""
        try:
            resp = llm_text.invoke(prompt)
            return resp.content.strip()
        except Exception:
            return chunk



    def _extract_metadata(self, text: str) -> Dict:
        """Genera Summary e Keywords."""
        try:
            resp = llm_text.invoke(f"{METADATA_PROMPT}\n{text[:2000]}")
            content = resp.content.strip()
            # Pulizia per garantire JSON valido
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content)
        except Exception:
            return {"summary": "N/A", "keywords": [], "topic": "Generico"}

    def process_document(self, transformed_data: Dict) -> List[Dict]:
        """
        Input: Output del Layer 3
        Output: Lista di chunk pronti per Layer 5/6/7
        """
        print("DEBUG content type:", type(transformed_data.get("content")))

        results: List[Dict] = []

        base_meta = transformed_data.get("metadata", {})
        doc_type = transformed_data.get("doc_type", "text")
        strategy = transformed_data.get("strategy_used", "text_extract")

        parent_doc_id = base_meta.get("sha256", str(uuid.uuid4()))
        content_type = transformed_data.get("content_type")

        print(f"⚙️ Rewriting Document: {base_meta.get('filename')} (Strategy: {strategy})")
        if transformed_data.get("content_type") == "image_list":
            image_paths = transformed_data.get("image_paths", [])
            for i, img_path in enumerate(image_paths):
                page_number = i + 1
                description = self._analyze_image_vision(img_path)
                if not isinstance(description, str) or not description.strip():
                    continue

                rewritten = self._rewrite_text_chunk(description, doc_type) or description

                norm = normalize_concepts(rewritten)
                canonical_concepts = norm["canonical_concepts"]
                semantic_meta = self._extract_metadata(rewritten)
                extracted_links = extract_links_from_text(rewritten)
                source_url = base_meta.get("source_url", "")
                if source_url and not any(l["url"] == source_url for l in extracted_links):
                    extracted_links.insert(0, {
                        "url": source_url,
                        "type": "url",
                        "description": f"Pagina web sorgente: {base_meta.get('filename', source_url)}"
                    })
                base_chunk_id = self._generate_chunk_id()

                results.append({
                    "chunk_id": base_chunk_id,
                    "parent_doc_id": parent_doc_id,
                    "text_content": rewritten,
                    "canonical_concepts": canonical_concepts,
                    "metadata": {
                        **base_meta,
                        **semantic_meta,
                        "chunk_index": i,
                        "page_number": page_number,
                        "source_type": "visual_guide",
                        "original_image": img_path,
                        "semantic_role": "descriptive",
                        "explicit": False,
                        "canonical_concepts": canonical_concepts,
                        "normalized_text": norm["normalized_text"],
                        "extracted_links": extracted_links,
                    }
                })

            return results

        normalized_pages = normalize_input_content(transformed_data)

        if not normalized_pages:
            return results

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", ", ", " "],
        )

        for page_index, page in enumerate(normalized_pages):
            page_text = page.get("text")
            page_number = page.get("page_number")

            if not isinstance(page_text, str) or not page_text.strip():
                continue

            docs = splitter.create_documents([page_text])

            for i, d in enumerate(docs):
                rewritten = self._rewrite_text_chunk(d.page_content, doc_type)
                if not rewritten:
                    continue

                norm = normalize_concepts(rewritten)
                canonical_concepts = norm["canonical_concepts"]

                semantic_meta = self._extract_metadata(rewritten)
                extracted_links = extract_links_from_text(rewritten)
                source_url = base_meta.get("source_url", "")
                if source_url and not any(l["url"] == source_url for l in extracted_links):
                    extracted_links.insert(0, {
                        "url": source_url,
                        "type": "url",
                        "description": f"Pagina web sorgente: {base_meta.get('filename', source_url)}"
                    })
                base_chunk_id = self._generate_chunk_id()

                base_chunk = {
                    "chunk_id": base_chunk_id,
                    "parent_doc_id": parent_doc_id,
                    "text_content": rewritten,
                    "canonical_concepts": canonical_concepts,
                    "metadata": {
                        **base_meta,
                        **semantic_meta,
                        "chunk_index": i,
                        "page_number": page_number,
                        "semantic_role": "descriptive",
                        "explicit": False,
                        "canonical_concepts": canonical_concepts,
                        "normalized_text": norm["normalized_text"],
                        "extracted_links": extracted_links,
                    }
                }

                results.append(base_chunk)
        return results

          


if __name__ == "__main__":
    # Test Standalone
    print("🧠 Testing Layer 4 (Final)...")
    from core.layers.layer1_data_collection import collect_documents # Nota il nome aggiornato
    from core.layers.layer2_quality_detection import detect_profile
    from core.layers.layer3_cleaning_transformation import transform_document

    rewriter = SemanticRewriter()
    docs = collect_documents()
    
    if docs:
        # Prendiamo il primo doc (sperando sia quello visuale per testare Vision)
        target = docs[0] 
        prof = detect_profile(target)
        transf = transform_document(prof)
        
        # Passiamo metadati base necessari
        transf["metadata"] = {"filename": target.filename, "sha256": target.sha256}
        transf["doc_type"] = prof.doc_type # Importante passare il doc_type rilevato
        
        final_chunks = rewriter.process_document(transf)
        
        print(f"\n✅ Generati {len(final_chunks)} chunks.")
        if final_chunks:
            sample = final_chunks[0]
            print(f"--- Sample Chunk [{sample['chunk_id']}] ---")
            print(f"Keywords: {sample['metadata'].get('keywords')}")
            print(f"Content Preview: {sample['text_content'][:200]}...")
            print("------------------------------------------")
    else:
        print("Nessun documento trovato.")