#!/usr/bin/env python3
"""
Runner per validazione del dataset validation_datasets/gold_questions.jsonl
Usa core.layers.layer8_application.process_request come backend per le query.
Esegui dalla root del repository (quella che contiene validation/ e validation_datasets/).
"""
import json
import os
import sys
from pathlib import Path

# Assicura che la root del repo sia nel PYTHONPATH per import relativi
repo_root = Path(__file__).resolve().parents[1]  # parent of validation/ -> repo root
sys.path.insert(0, str(repo_root))

# Import del validatore (package local)
from validation.zero_knowledge_validator import ZeroKnowledgeValidator

# Import della funzione process_request da layer8_application
try:
    from core.layers.layer8_application import process_request
except Exception as e:
    raise RuntimeError(f"Impossibile importare core.layers.layer8_application.process_request: {e}")

def chatbot_query_func(question: str):
    """
    Wrapper che chiama process_request(question) e normalizza l'output nella shape attesa:
    {'answer': str, 'chunks': List[str], 'metadata': List[Dict]}
    """
    # process_request signature: process_request(question: str, chat_history: List[Dict] = None) -> Dict
    res = process_request(question)
    # estrai answer
    answer = res.get("answer") or res.get("answer_main") or res.get("output") or ""
    # possibili nomi dei contesti recuperati
    chunks = res.get("retrieved_contexts") or res.get("retrieved_chunks") or res.get("contexts") or []
    chunks_metadata = res.get("chunks_metadata") or res.get("chunks_meta") or res.get("sources") or []
    # normalizza chunk in stringhe
    normalized_chunks = []
    for c in chunks:
        if isinstance(c, str):
            normalized_chunks.append(c)
        elif isinstance(c, dict):
            normalized_chunks.append(c.get("page_content") or c.get("text") or str(c))
        else:
            normalized_chunks.append(str(c))
    return {"answer": answer, "chunks": normalized_chunks, "metadata": chunks_metadata}

def load_questions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")
    qs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id") or obj.get("id")
            qtext = obj.get("user_input") or obj.get("question") or obj.get("text")
            if not qtext:
                raise RuntimeError(f"Domanda mancante in record: {obj}")
            qs.append({"question_id": qid, "question": qtext})
    return qs

def main():
    repo_root = Path.cwd()
    dataset_path = repo_root / "validation_datasets" / "gold_questions.jsonl"
    questions = load_questions(dataset_path)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ OPENAI_API_KEY non impostata nell'ambiente. Impostala prima di eseguire.")
        # si può scegliere di proseguire, ma le chiamate LLM falliranno
    validator = ZeroKnowledgeValidator(
        chatbot_query_func=chatbot_query_func,
        openai_api_key=openai_key,
        output_dir="validation_results",
        num_consistency_runs=3
    )

    final_report = validator.validate_dataset(questions, test_name="gold_questions_zero_knowledge")
    print("Validazione completata. Report salvato in: validation_results/")
    print("Report name:", final_report.get("test_name"))

if __name__ == "__main__":
    main()