"""
ragas_eval.py

Runs RAGAS (Retrieval Augmented Generation Assessment) evaluation on the RAG system.
It uses the 'process_request' function from layer8_application.py to generate answers and contexts,
and then computes metrics using the RAGAS library.

Requirements:
  pip install ragas datasets pandas openpyxl

Usage:
  python validation/ragas_eval.py --dataset validation/gold_dataset.jsonl --out results/ragas_report.csv
"""

import argparse
import json
import sys
import os
import pandas as pd
import types
import time
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
import re
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- 1. Environment Setup ---
# Resolve project root to import core modules
current = Path(__file__).resolve()
project_root = None
for p in current.parents:
    if (p / ".env").exists():
        project_root = p
        break
if project_root is None:
    raise RuntimeError("Impossibile trovare .env risalendo dalle parent dirs")

sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

# --- PATCH: Fix for langchain.schema missing in newer versions ---
try:
    import langchain.schema  # type: ignore
except ImportError:
    try:
        import langchain  # type: ignore
        import langchain_core.documents  # type: ignore
        m = types.ModuleType("langchain.schema")
        m.Document = langchain_core.documents.Document  # type: ignore
        sys.modules["langchain.schema"] = m
        langchain.schema = m
        print("🔧 Patched 'langchain.schema' using 'langchain_core'")
    except ImportError as e:
        print(f"⚠️ Failed to patch langchain.schema: {e}")
        pass

# --- 2. Conditional Imports ---
try:
    # Debug import retriever
    import core.custom.ensemble_retriever as er
    print("ENSEMBLE RETRIEVER LOADED FROM:", er.__file__)

    from core.layers.layer8_application import process_request
except ImportError as e:
    print(f"❌ ERROR: Could not import 'core.layers.layer8_application'.\n   Details: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from datasets import Dataset
    from ragas import evaluate
    # Aggiornamento import come richiesto (compatibilità versioni recenti)
    try:
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
    except ImportError:
        from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness

except ImportError as e:
    print(f"❌ ERROR: Could not import 'ragas' or 'datasets'.\n   Details: {e}")
    if "PIL" in str(e) or "_imaging" in str(e):
        print("   ⚠️  Pillow library seems corrupted. Run: pip install --force-reinstall Pillow")
    else:
        print("   Install with: pip install ragas datasets pandas openpyxl")
    sys.exit(1)

# --- 3. Load Configuration from Environment (Same as Layer 8) ---
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = "text-embedding-3-small"

print(f"🤖 Using LLM Model: {LLM_MODEL}")
print(f"🧩 Using Embedding Model: {EMBED_MODEL}")

def load_dataset(path: Path) -> list:
    """Loads JSONL dataset."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def check_factual_consistency(answer: str, ground_truth: str) -> str:
    """Verifica presenza di email/numeri chiave."""
    # Regex per email
    emails_gt = set(re.findall(r'[\w\.-]+@[\w\.-]+', ground_truth))
    emails_ans = set(re.findall(r'[\w\.-]+@[\w\.-]+', answer))
    
    if emails_gt and not emails_gt.issubset(emails_ans):
        return "EMAIL_MISMATCH"
    return "OK"

def main():
    parser = argparse.ArgumentParser(description="RAGAS Evaluation for ClassyFarm RAG")
    parser.add_argument("--dataset", type=Path, default=project_root / "validation/gold_dataset.jsonl", 
                        help="Path to JSONL dataset (default: validation/gold_dataset.jsonl)")
    parser.add_argument("--out", type=Path, default=Path("results/ragas_report.csv"), 
                        help="Output CSV path")
    parser.add_argument("--limit", type=int, default=0, 
                        help="Max samples to evaluate (0=all)")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"❌ Dataset file not found: {args.dataset}")
        sys.exit(1)

    # 1. Load Data
    raw_data = load_dataset(args.dataset)
    if args.limit > 0:
        raw_data = raw_data[:args.limit]
    
    print(f"🔹 Loaded dataset: {len(raw_data)} samples.")

    # 2. Prepare Lists for RAGAS
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    metadatas = []
    # Nuove metriche logiche
    factual_checks = []
    answerable_flags = []
    
    print("🚀 Running RAG Inference...")
    for item in tqdm(raw_data, desc="Inference"):
        q = item.get("question")
        if not q: continue

        # Retrieve ground truth (supports various field names)
        gt = item.get("ground_truth") or item.get("answer") or item.get("expected_answer") or ""
        
        try:
            # Call your RAG Engine
            # NOTE: process_request() in layer8_application.py doesn't accept 'role' parameter
            res = process_request(question=q)
            
            # Extract data
            ans = res.get("answer", "")
            # layer8_application returns 'retrieved_contexts' as list[str], which is what RAGAS needs
            ctxs = res.get("retrieved_contexts", [])
            
            questions.append(q)
            answers.append(ans)
            contexts.append(ctxs)
            # RAGAS v0.2+ expects 'ground_truth' to be a single string
            ground_truths.append(gt)
            metadatas.append(item.get("metadata", {}))
            
            # Post-processing logico
            factual_checks.append(check_factual_consistency(ans, gt))
            
            # Euristica: se il contesto è vuoto, non era answerable dai documenti recuperati
            is_answerable = len(ctxs) > 0 and "non ho trovato" not in ans.lower()
            answerable_flags.append(is_answerable)
            
            # Delay per evitare rate limiting durante inference
            time.sleep(0.2)  # 200ms di pausa tra richieste
            
        except Exception as e:
            print(f"⚠️ Error on '{q}': {e}")
            # Se l'errore è strutturale (init fallita), inutile continuare per tutto il dataset
            if "init Retrievers" in str(e) or "object has no attribute 'add'" in str(e):
                print("\n🛑 CRITICAL ERROR: Il motore RAG non riesce ad inizializzarsi.")
                print("   Verifica 'core/custom/ensemble_retriever.py' o 'layer8_application.py' per errori di tipo {}.add()")
                sys.exit(1)
            continue

    # 3. Create HuggingFace Dataset for RAGAS
    if len(questions) == 0:
        print("❌ No samples generated for evaluation. Check RAG Engine errors above.")
        sys.exit(1)

    ragas_data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
        "metadata": metadatas,
        # Campi custom per analisi (non usati da Ragas ma utili nel CSV)
        "factual_mismatch": factual_checks
    }
    dataset = Dataset.from_dict(ragas_data)

    print(f"✅ Generated {len(dataset)} samples for evaluation")

    # 4. Configure RAGAS Evaluator LLM & Embeddings
    # Use same models as Layer 8 for consistency
    print("\n🔧 Configuring RAGAS Evaluator...")
    print(f"   LLM: {LLM_MODEL}")
    print(f"   Embeddings: {EMBED_MODEL}")
    
    evaluator_llm = ChatOpenAI(
        model=LLM_MODEL,  # Same as Layer 8
        temperature=0.0,
        request_timeout=120,  # 2 minutes timeout
        max_retries=3,
        max_tokens=4096
    )
    
    evaluator_embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,  # Same as Layer 8
        request_timeout=90
    )

    # 5. Configure Metrics
    # Note: context_recall and context_precision rely heavily on the quality of ground_truth
    metrics = [
        faithfulness,      # Is the answer derived from the context?
        answer_relevancy,  # Is the answer relevant to the question?
        context_precision, # Are relevant chunks ranked higher?
        context_recall,    # Does the retrieved context cover the ground truth?
        answer_correctness # Does the answer match the ground truth semantically?
    ]

    print(f"\n🧐 Starting RAGAS evaluation on {len(dataset)} samples...")
    print("   (This involves calls to OpenAI API for evaluation)")
    print("   ⏳ This may take several minutes depending on dataset size...")

    # 6. Run Evaluation with Error Handling
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            raise_exceptions=False,  # Continue even if some evaluations fail
            # max_workers=2  # Uncomment to reduce parallelism if hitting rate limits
        )

        print("\n=== 📊 RAGAS RESULTS ===")
        print(results)

    except Exception as e:
        print(f"\n❌ RAGAS Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 7. Export Results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Aggiunge timestamp al nome file per non sovrascrivere i risultati precedenti
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.out.with_name(f"{args.out.stem}_{timestamp}{args.out.suffix}")

    df = results.to_pandas()
    
    # Aggiungi colonne custom
    df['factual_mismatch'] = factual_checks[:len(df)]
    df['answerable'] = answerable_flags[:len(df)]
    
    df.to_csv(output_path, index=False)
    print(f"\n✅ Report saved to: {output_path}")
    
    # 8. Print Summary Statistics
    print("\n=== 📈 SUMMARY STATISTICS ===")
    print(f"Total Samples: {len(df)}")
    print(f"\nMetric Averages:")
    for col in df.columns:
        if col not in ['question', 'answer', 'contexts', 'ground_truth', 'metadata', 'factual_mismatch', 'answerable']:
            try:
                mean_val = df[col].mean()
                print(f"  {col}: {mean_val:.3f}")
            except:
                pass
    
    print(f"\nFactual Mismatches: {sum(1 for x in factual_checks if x != 'OK')}/{len(factual_checks)}")
    print(f"Answerable Questions: {sum(answerable_flags)}/{len(answerable_flags)}")

if __name__ == "__main__":
    main()