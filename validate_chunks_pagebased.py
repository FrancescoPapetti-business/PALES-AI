import json, re
from pathlib import Path

import fitz  # PyMuPDF
from rapidfuzz import fuzz

CHUNKS = Path(r"data\datasets\v1_20251216_170818\chunks.jsonl.bak")
PDF_ROOT = Path(r"data\pdf_input")
OUT = Path(r"data\datasets\v1_20251216_170818\chunk_validation_report.jsonl")

ws = re.compile(r"\s+")
punct = re.compile(r"[^\w\s%€°àèìòùÀÈÌÒÙ\-]")
num = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\b")
date = re.compile(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b")


def norm(s: str) -> str:
    s = (s or "").lower().replace("\u00ad", "")
    s = punct.sub(" ", s)
    return ws.sub(" ", s).strip()


def page_text(pdf_path: Path, page_1based: int) -> str:
    d = fitz.open(pdf_path)
    try:
        i = int(page_1based) - 1
        if i < 0 or i >= d.page_count:
            return ""
        return d.load_page(i).get_text("text") or ""
    finally:
        d.close()


def tok_overlap(a: str, b: str) -> float:
    A, B = set(a.split()), set(b.split())
    return 0.0 if not A else len(A & B) / len(A)


def extract_nums(s: str):
    # filtra numeri “spazzatura” di impaginazione (pag/pagina/p.)
    s = (s or "").lower()
    s = re.sub(r"\bpag(?:ina)?\b\s*\d{1,4}\b", " ", s)
    s = re.sub(r"\bp\.\s*\d{1,4}\b", " ", s)
    return set(num.findall(s)) | set(date.findall(s))


def best_window(page_txt: str, chunk_norm: str, win_lines: int = 5):
    # usa token_sort_ratio (non token_set_ratio) per evitare 100 su sottoinsiemi
    lines = [norm(x) for x in (page_txt or "").splitlines() if norm(x)]
    if not lines:
        return "", 0

    best_txt, best_score = "", 0
    for i in range(len(lines)):
        w = " ".join(lines[i:i + win_lines])
        s = fuzz.token_sort_ratio(chunk_norm, w)
        if s > best_score:
            best_score = s
            best_txt = w
    return best_txt, best_score

def best_page_window(pdf_path: Path, page_1based: int, chunk_norm: str, win_lines: int = 5):
    # prova page-1, page, page+1 e prende la migliore
    candidates = []
    for delta in (-1, 0, 1):
        p = int(page_1based) + delta
        if p <= 0:
            continue
        txt = page_text(pdf_path, p)
        w, s = best_window(txt, chunk_norm, win_lines=win_lines)
        candidates.append((s, p, w))
    if not candidates:
        return 0, page_1based, ""
    candidates.sort(reverse=True, key=lambda x: x[0])
    best_s, best_p, best_w = candidates[0]
    return best_s, best_p, best_w

OUT.parent.mkdir(parents=True, exist_ok=True)

with CHUNKS.open("r", encoding="utf-8") as fi, OUT.open("w", encoding="utf-8") as fo:
    for line in fi:
        line = line.strip()
        if not line:
            continue

        c = json.loads(line)
        m = c.get("metadata", {}) or {}

        rel = (m.get("rel_path") or "").replace("\\", "/")
        pdf = (PDF_ROOT / rel)
        page = m.get("page_number")

        chunk_txt = c.get("text_content") or ""

        a = norm(chunk_txt)

        if pdf.exists() and page:
            score, best_page, b_window = best_page_window(pdf, page, a, win_lines=5)
        else:
            score, best_page, b_window = 0, page, ""

        overlap = tok_overlap(a, b_window) if b_window else 0.0

        # numeri/date: solo sulla finestra migliore
        mismatch_nums = bool(extract_nums(a) - extract_nums(b_window)) if b_window else True

        if not pdf.exists() or not page:
            status, err = "INVALID", "mapping_error"
        elif not b_window:
            status, err = "INVALID", "no_text_extracted"
        elif score >= 92 and overlap >= 0.65 and not mismatch_nums:
            status, err = "VERIFIED", "none"
        elif score >= 80 and overlap >= 0.45 and not mismatch_nums:
            status, err = "NEEDS_REVIEW", "weak_alignment"
        elif mismatch_nums:
            status, err = "NEEDS_REVIEW", "numeric_mismatch"
        else:
            status, err = "INVALID", "text_mismatch"


        fo.write(json.dumps({
            "chunk_id": c.get("chunk_id"),
            "filename": m.get("filename"),
            "rel_path": m.get("rel_path"),
            "page_number": page,
            "best_page_number": best_page,
            "pdf_resolved_path": str(pdf),
            "status": status,
            "score": int(score),
            "token_overlap": round(overlap, 4),
            "error_type": err
        }, ensure_ascii=False) + "\n")

print("OK:", OUT)
