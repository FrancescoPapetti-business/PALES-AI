import pandas as pd
import json
import os
from tqdm import tqdm
from pathlib import Path

# Import locali
from tools.validation.utils import load_config, load_dataset_jsonl, safe_post
from tools.validation.metrics import (
    recall_at_k, precision_at_k, 
    ood_detection_score, injection_resistance_score, legal_safety_score
)

def extract_ids_from_response(resp: dict) -> list:
    """Estrae ID robustamente indipendentemente dal formato API."""
    # 1. Formato Custom esplicito
    if "retrieved_chunks_ids" in resp:
        return resp["retrieved_chunks_ids"]
    
    # 2. Formato LangChain standard (source_documents)
    ids = []
    if "source_documents" in resp:
        for doc in resp["source_documents"]:
            # Cerca ID nei metadata
            meta = doc.get("metadata", {})
            # Priorità: chunk_id > id > source
            found = meta.get("chunk_id") or meta.get("id") or meta.get("source")
            if found:
                ids.append(str(found))
    return ids

def run_validation():
    print("🚀 Avvio Validazione...")
    try:
        config = load_config()
    except Exception as e:
        print(f"Errore caricamento config: {str(e)}")
        return

    dataset = load_dataset_jsonl(config["dataset_path"])
    if not dataset:
        print("❌ Dataset vuoto. Esco.")
        return

    # Setup
    api_url = config["api_url"]
    max_items = config.get("max_items")
    if max_items:
        dataset = dataset[:max_items]

    results = []

    for item in tqdm(dataset, desc="Processing"):
        question = item.get("question")
        expected_chunks = [str(x) for x in item.get("expected_chunks", [])] # Force string check

        # Chiamata API
        payload = {"message": question}
        resp = safe_post(api_url, payload, timeout=config.get("request_timeout", 180))

        row = {
            "id": item.get("id"),
            "question": question,
            "error": None
        }

        if resp:
            answer = resp.get("answer", "") or resp.get("response", "")
            retrieved_ids = extract_ids_from_response(resp)

            row.update({
                "answer": answer,
                "retrieved_ids": retrieved_ids,
                "recall@5": recall_at_k(retrieved_ids, expected_chunks, k=5),
                "precision@5": precision_at_k(retrieved_ids, expected_chunks, k=5),
                "OOD_score": ood_detection_score(question, answer),
                "injection_score": injection_resistance_score(question, answer),
                "legal_score": legal_safety_score(answer)
            })
        else:
            row["error"] = "API Failed"
            row.update({
                "answer": None,
                "retrieved_ids": [],
                "recall@5": 0.0,
                "precision@5": 0.0,
                "OOD_score": 0,
                "injection_score": 0,
                "legal_score": 0
            })
        results.append(row)

    # Salvataggio
    df = pd.DataFrame(results)
    
    # Gestione Path Assoluti per output
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Root del progetto
    raw_path = os.path.join(base_dir, config["results_raw_path"])
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    df.to_csv(raw_path, index=False)
    summary = {
        "n_items": len(df),
        "recall@5_mean": df["recall@5"].mean() if not df.empty else 0.0,
        "precision@5_mean": df["precision@5"].mean() if not df.empty else 0.0,
        "OOD_mean": df["OOD_score"].mean() if not df.empty else 0.0,
        "injection_resistance_mean": df["injection_score"].mean() if not df.empty else 0.0,
        "legal_safety_mean": df["legal_score"].mean() if not df.empty else 0.0
    }

    summary_path = os.path.join(base_dir, config["results_aggregated_path"])
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

        
    print(f"\n✅ Validazione completata.")
    print(f"📊 Recall Media: {df['recall@5'].mean():.2f}")
    print(f"📂 Output: {raw_path}")

if __name__ == "__main__":
    run_validation()
