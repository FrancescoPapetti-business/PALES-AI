# validate_chunks_vision_judge.py
# Multimodal Vision-as-a-Judge: confronta immagine pagina PDF vs chunk (testo) e produce report JSONL
#
# Requisiti:
#   pip install pymupdf rapidfuzz requests
#
# Uso (da root progetto classyfarm-rag):
#   set OPENAI_API_KEY=...
#   python validate_chunks_vision_judge.py ^
#     --chunks data\datasets\v1_20251216_170818\chunks.jsonl.bak ^
#     --pdf-root data\pdf_input ^
#     --out data\datasets\v1_20251216_170818\chunk_validation_vision_report.jsonl ^
#     --model gpt-4.1 ^
#     --dpi 200 ^
#     --max-pages-shift 1 ^
#     --limit 0
#
# Note:
# - max-pages-shift=1 prova anche pagina-1 e pagina+1 (utile se page_number è off-by-one).
# - limit=0 significa “tutti”.
from dotenv import load_dotenv
load_dotenv()
import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import fitz  # PyMuPDF
import requests

# -------------------------
# Prompt / schema di output
# -------------------------
JUDGE_SYSTEM = """Sei un revisore QA per un sistema RAG industriale.
Valuti la coerenza tra una pagina di manuale (immagine) e un chunk di testo estratto da quella pagina.
Devi essere severo su numeri/codici/sigle e non inventare nulla.
Rispondi SOLO con JSON valido, senza testo extra."""

JUDGE_USER_TEMPLATE = """INPUT:
1) IMMAGINE: rendering della pagina del PDF.
2) CHUNK_TESTO: testo estratto (potenzialmente generato da descrizione visiva).

COMPITO:
Valuta se CHUNK_TESTO rappresenta fedelmente il contenuto visivo della pagina.

METRICHE (obbligatorie):
- visual_faithfulness (1-5): accuratezza della descrizione di elementi visivi (schermate UI, tabelle, diagrammi, pulsanti, campi, flussi).
- procedural_integrity ("pass"|"fail"): se il chunk descrive una procedura, verifica ordine e completezza dei passaggi visibili.
- data_accuracy ("pass"|"fail"): numeri, date, codici, sigle, label UI devono corrispondere. Se manca anche un solo elemento critico -> fail.
- contextual_noise (0-100): percentuale stimata del chunk che è rumore / fuori contesto rispetto alla pagina.
- quality_score (1-5): valutazione complessiva.

REGOLE DI VALIDITÀ:
- valid = true SOLO SE:
  (visual_faithfulness >= 4) AND (procedural_integrity == "pass") AND (data_accuracy == "pass") AND (contextual_noise <= 20)

CLASSIFICAZIONE ERRORI:
error_type deve essere uno tra:
"none" | "hallucination" | "omission" | "data_mismatch" | "wrong_page" | "garbled_text" | "layout_misread"

OUTPUT (JSON ESATTO):
{
  "valid": boolean,
  "visual_faithfulness": 1-5,
  "procedural_integrity": "pass"|"fail",
  "data_accuracy": "pass"|"fail",
  "contextual_noise": 0-100,
  "quality_score": 1-5,
  "error_type": "...",
  "justification": "massimo 2 frasi, concreta"
}

CHUNK_TESTO:
<<<{chunk_text}>>>
"""

JSON_RE = re.compile(r"\{.*\}\s*$", re.DOTALL)

def clamp_int(x, lo, hi, default):
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default

def render_page_to_png_bytes(pdf_path: Path, page_1based: int, dpi: int) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        i = int(page_1based) - 1
        if i < 0 or i >= doc.page_count:
            return b""
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()

def b64_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"

def call_openai_vision(model: str, api_key: str, img_data_url: str, user_text: str, timeout_s: int, max_retries: int = 3):
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": JUDGE_SYSTEM}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": img_data_url},
                    {"type": "input_text", "text": user_text},
                ],
            },
        ],
        # spingi JSON “pulito”
        "text": {"format": {"type": "json_object"}},
        "temperature": 0,
    }

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
            if r.status_code == 429 or 500 <= r.status_code <= 599:
                last_err = f"http_{r.status_code}"
                time.sleep(min(10, 2 * attempt))
                continue
            r.raise_for_status()
            data = r.json()

            # Estrazione testo: Responses API ritorna tipicamente output_text aggregato
            out_text = data.get("output_text")
            if not out_text:
                # fallback: cerca in output[...].content[...].text
                out = data.get("output", [])
                texts = []
                for item in out:
                    for c in item.get("content", []):
                        t = c.get("text")
                        if t:
                            texts.append(t)
                out_text = "\n".join(texts).strip()

            if not out_text:
                raise RuntimeError("empty_model_output")

            # garantisci JSON puro
            m = JSON_RE.search(out_text.strip())
            json_str = m.group(0) if m else out_text.strip()
            obj = json.loads(json_str)
            return obj, None
        except Exception as e:
            last_err = str(e)
            time.sleep(min(10, 2 * attempt))
    return None, last_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=r"data\datasets\v1_20251216_170818\chunks_v2.jsonl.bak", type=str)
    ap.add_argument("--pdf-root", default=r"data\pdf_input", type=str)
    ap.add_argument("--out", default=r"data\datasets\v1_20251216_170818\chunk_validation_vision_report_v2.jsonl", type=str)
    ap.add_argument("--model", default="gpt-4.1", type=str)
    ap.add_argument("--dpi", default=200, type=int)
    ap.add_argument("--timeout", default=120, type=int)
    ap.add_argument("--max-pages-shift", default=1, type=int)  # 0=solo pagina indicata, 1=anche -1 +1, ecc.
    ap.add_argument("--limit", default=0, type=int)  # 0=tutti
    ap.add_argument("--sleep", default=0.0, type=float)  # pausa tra chiamate
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ERRORE: OPENAI_API_KEY non impostata nell'ambiente.")

    chunks_path = Path(args.chunks)
    pdf_root = Path(args.pdf_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = max(120, int(args.dpi))
    max_shift = max(0, int(args.max_pages_shift))
    limit = int(args.limit)

    n = 0
    with chunks_path.open("r", encoding="utf-8") as fi, out_path.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue

            c = json.loads(line)
            m = c.get("metadata", {}) or {}

            rel = (m.get("rel_path") or "").replace("\\", "/").strip()
            pdf_path = (pdf_root / rel)
            page = m.get("page_number")

            chunk_id = c.get("chunk_id")
            chunk_txt = c.get("text_content") or ""

            record = {
                "chunk_id": chunk_id,
                "filename": m.get("filename"),
                "rel_path": m.get("rel_path"),
                "page_number": page,
                "pdf_resolved_path": str(pdf_path),
                "model": args.model,
                "dpi": dpi,
            }

            if not pdf_path.exists() or not page:
                record.update({
                    "status": "INVALID",
                    "error_type": "mapping_error",
                    "judge": None,
                    "best_page_number": None,
                    "note": "pdf_missing_or_page_missing",
                })
                fo.write(json.dumps(record, ensure_ascii=False) + "\n")
                n += 1
                if limit and n >= limit:
                    break
                continue

            # prova pagina indicata e, se richiesto, +/- shift per robustezza
            page_int = int(page)
            candidates = [page_int]
            for s in range(1, max_shift + 1):
                candidates.append(page_int - s)
                candidates.append(page_int + s)

            best = None  # (valid, quality_score, chosen_page, judge_obj)
            best_err = None

            for p in candidates:
                if p <= 0:
                    continue
                png_bytes = render_page_to_png_bytes(pdf_path, p, dpi=dpi)
                if not png_bytes:
                    continue

                user_prompt = JUDGE_USER_TEMPLATE.replace("{chunk_text}", chunk_txt)
                judge_obj, err = call_openai_vision(
                    model=args.model,
                    api_key=api_key,
                    img_data_url=b64_data_url(png_bytes),
                    user_text=user_prompt,
                    timeout_s=args.timeout,
                )
                if err:
                    best_err = err
                    continue

                # normalizza output atteso
                valid = bool(judge_obj.get("valid", False))
                q = clamp_int(judge_obj.get("quality_score", 0), 1, 5, 1)
                vf = clamp_int(judge_obj.get("visual_faithfulness", 0), 1, 5, 1)
                noise = clamp_int(judge_obj.get("contextual_noise", 100), 0, 100, 100)
                pi = judge_obj.get("procedural_integrity", "fail")
                da = judge_obj.get("data_accuracy", "fail")

                # score di selezione candidato: preferisci valid, poi quality, poi faithfulness, poi meno noise
                sel = (1 if valid else 0, q, vf, -noise)

                if best is None or sel > best[0]:
                    best = (sel, p, judge_obj)

                # se già valido “forte”, stop early
                if valid and q == 5 and vf == 5 and noise <= 10 and pi == "pass" and da == "pass":
                    break

                if args.sleep > 0:
                    time.sleep(args.sleep)

            if best is None:
                record.update({
                    "status": "INVALID",
                    "error_type": "judge_call_failed",
                    "judge": None,
                    "best_page_number": None,
                    "note": best_err or "no_candidate_pages_rendered",
                })
            else:
                _, chosen_page, judge_obj = best

                # status finale
                status = "VERIFIED" if judge_obj.get("valid") else "NEEDS_REVIEW"
                err_type = judge_obj.get("error_type") or ("none" if judge_obj.get("valid") else "unknown")

                record.update({
                    "best_page_number": chosen_page,
                    "status": status,
                    "error_type": err_type,
                    "judge": judge_obj,
                })

            fo.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
            if limit and n >= limit:
                break

    print("OK:", out_path)

if __name__ == "__main__":
    main()
