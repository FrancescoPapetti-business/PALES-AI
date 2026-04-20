"""
Script per eseguire validazione zero-knowledge su ClassyFarm RAG
"""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path per import relativi
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import validator

from validation.zero_knowledge_validator import ZeroKnowledgeValidator

def load_gold_dataset(dataset_path: str = "validation_datasets/gold_questions.jsonl"):
    """Carica dataset domande da JSONL"""
    questions = []
    
    dataset_full_path = Path(__file__).parent.parent / dataset_path
    
    if not dataset_full_path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {dataset_full_path}")
    
    with open(dataset_full_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                questions.append({
                    'question_id': data.get('question_id', data.get('user_input')),
                    'question': data.get('user_input', data.get('question'))
                })
    
    return questions


def chatbot_query_wrapper(question: str) -> dict:
    from core.layers.layer8_application import process_request
    
    # Chiama SENZA role
    response = process_request(question=question)
    
    chunks = response.get('retrieved_contexts', [])
    sources = response.get('sources', [])
    
    metadata = []
    for source in sources:
        if isinstance(source, dict):
            metadata.append(source)
        elif hasattr(source, 'metadata'):
            metadata.append(source.metadata)
        else:
            metadata.append({'source': str(source)})
    
    if not metadata and chunks:
        metadata = [{'source': 'documento', 'index': i} for i in range(len(chunks))]
    
    return {
        'answer': response.get('answer', ''),
        'chunks': chunks,
        'metadata': metadata
    }


def main():
    """Entry point validazione"""
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY non trovata. Assicurati di averla nel file .env:\n"
            "OPENAI_API_KEY=sk-..."
        )
    
    print("\n" + "="*70)
    print(" "*15 + "VALIDAZIONE ZERO-KNOWLEDGE CLASSYFARM-RAG")
    print("="*70 + "\n")
    
    # Carica dataset domande
    print("📂 Caricamento gold dataset...")
    try:
        questions = load_gold_dataset("validation_datasets/gold_questions.jsonl")
        print(f"   ✓ Caricate {len(questions)} domande\n")
    except FileNotFoundError as e:
        print(f"   ✗ ERRORE: {e}")
        print("\n   Crea il file con:")
        print("   mkdir -p validation_datasets")
        print("   # Poi popola gold_questions.jsonl (vedi formato sotto)\n")
        return
    
    # Inizializza validator
    print("🔧 Inizializzazione validator...")
    validator = ZeroKnowledgeValidator(
        chatbot_query_func=chatbot_query_wrapper,
        openai_api_key=openai_api_key,
        output_dir="validation_results",
        num_consistency_runs=3,  # 3 run per calcolare consistency
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model="gpt-4o-mini"  # Usa GPT-4o-mini (più economico di GPT-4)
    )
    print("   ✓ Validator pronto\n")
    
    # Conferma prima di procedere
    print(f"⚠️  Stai per eseguire validazione su {len(questions)} domande")
    print(f"    con {validator.num_consistency_runs} run ciascuna")
    print(f"    = {len(questions) * validator.num_consistency_runs} chiamate al chatbot totali\n")
    
    proceed = input("Procedere? [y/N]: ").strip().lower()
    if proceed != 'y':
        print("Validazione annullata.")
        return
    
    # Esegui validazione completa
    print("\n🚀 Avvio validazione...\n")
    report = validator.validate_dataset(
        questions=questions,
        test_name=None  # Auto-genera nome con timestamp YYYYMMDD_HHMMSS
    )
    
    # Summary finale
    print("\n" + "="*70)
    print(" "*25 + "✓ VALIDAZIONE COMPLETATA")
    print("="*70)
    print(f"\n📊 Health Score: {report['health_score']:.1f}%")
    print(f"📁 Risultati salvati in: validation_results/")
    print(f"   - JSON: {report['test_name']}.json")
    print(f"   - CSV:  {report['test_name']}.csv\n")
    
    # Top 3 domande critiche
    if report['critical_questions']:
        print("🚨 Top 3 domande critiche:")
        for i, q in enumerate(report['critical_questions'][:3], 1):
            print(f"   {i}. {q['question_id']}: {q['question'][:60]}...")
            print(f"      Flags: {q['critical_flags']} | Metriche fallite: {', '.join(q['failed_metrics'])}")
        print()


if __name__ == "__main__":
    main()