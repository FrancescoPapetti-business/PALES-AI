"""
Validatore zero-knowledge: validazione senza ground truth
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from .metrics_calculator import MetricsCalculator


class ZeroKnowledgeValidator:
    """
    Validatore completo senza bisogno di reference/ground truth
    """
    
    def __init__(
        self,
        chatbot_query_func,  # Funzione che prende domanda e ritorna (risposta, chunks, metadata)
        openai_api_key: str,
        output_dir: str = "validation_results",
        num_consistency_runs: int = 3,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Args:
            chatbot_query_func: Funzione callable(question: str) -> Dict con:
                - 'answer': str (risposta generata)
                - 'chunks': List[str] (documenti recuperati)
                - 'metadata': List[Dict] (metadata chunk con 'source', 'page', etc)
            openai_api_key: API key OpenAI per LLM-as-judge
            output_dir: Directory dove salvare risultati
            num_consistency_runs: Numero di run per consistency check (default 3)
        """
        self.chatbot_query = chatbot_query_func
        self.num_consistency_runs = num_consistency_runs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Inizializza calcolatore metriche
        self.metrics_calc = MetricsCalculator(
            openai_api_key=openai_api_key,
            embedding_model=embedding_model,
            llm_model=llm_model
        )
        
        # Thresholds per classificazione
        self.thresholds = {
            'groundedness': 0.85,
            'self_contained': 0.70,  # 3.5/5 normalizzato
            'relevance': 0.70,
            'consistency': 0.75,
            'context_precision': 0.60
        }
    
    def validate_single_question(
        self, 
        question: str,
        question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Valida una singola domanda con tutte le metriche
        
        Returns:
            Dict con tutte le metriche e dettagli
        """
        print(f"\n{'='*60}")
        print(f"Validazione: {question_id or 'N/A'}")
        print(f"Domanda: {question[:100]}...")
        print(f"{'='*60}")
        
        # ==== RUN 1: Risposta principale + metriche ====
        print("\n[1/5] Generazione risposta principale...")
        result_main = self.chatbot_query(question)
        answer_main = result_main['answer']
        chunks = result_main['chunks']
        chunks_metadata = result_main.get('metadata', [])
        
        print(f"  ✓ Risposta: {len(answer_main)} caratteri")
        print(f"  ✓ Chunk recuperati: {len(chunks)}")
        
        # ==== RUN 2-N: Consistency check ====
        print(f"\n[2/5] Generazione {self.num_consistency_runs-1} risposte aggiuntive per consistency...")
        additional_answers = [answer_main]
        
        for i in range(self.num_consistency_runs - 1):
            result_extra = self.chatbot_query(question)
            additional_answers.append(result_extra['answer'])
            print(f"  ✓ Risposta {i+2}: {len(result_extra['answer'])} caratteri")
        
        # ==== METRICA 1: Groundedness ====
        print("\n[3/5] Calcolo Groundedness...")
        groundedness, groundedness_details = self.metrics_calc.calculate_groundedness(
            answer_main, chunks
        )
        print(f"  ✓ Groundedness: {groundedness:.3f}")
        
        # ==== METRICA 2: Self-Contained ====
        print("\n[4/5] Calcolo Self-Contained Score...")
        self_contained, self_contained_reason = self.metrics_calc.calculate_self_contained(
            question, answer_main
        )
        print(f"  ✓ Self-Contained: {self_contained:.3f} ({self_contained:.1%} su scala 1-5)")
        
        # ==== METRICA 3: Relevance (embedding) ====
        print("\n[5/5] Calcolo Relevance...")
        relevance_emb = self.metrics_calc.calculate_relevance_embedding(
            question, answer_main
        )
        print(f"  ✓ Relevance (embedding): {relevance_emb:.3f}")
        
        # Opzionale: Relevance con LLM (più lento ma più interpretabile)
        # relevance_llm, relevance_status = self.metrics_calc.calculate_relevance_llm(
        #     question, answer_main
        # )
        
        # ==== METRICA 4: Consistency ====
        print("\n[6/5] Calcolo Consistency...")
        consistency, consistency_details = self.metrics_calc.calculate_consistency(
            additional_answers
        )
        print(f"  ✓ Consistency: {consistency:.3f}")
        
        # ==== METRICA 5: Context Precision ====
        print("\n[7/5] Calcolo Context Precision...")
        context_precision, chunk_evaluations = self.metrics_calc.calculate_context_precision(
            question, chunks
        )
        print(f"  ✓ Context Precision: {context_precision:.3f}")
        
        # ==== BONUS: Citation Accuracy ====
        print("\n[8/5] Calcolo Citation Accuracy...")
        citation_accuracy, citation_details = self.metrics_calc.calculate_citation_accuracy(
            answer_main, chunks_metadata
        )
        print(f"  ✓ Citation Accuracy: {citation_accuracy:.3f}")
        
        # ==== Classificazione critica ====
        critical_flags = 0
        failed_metrics = []
        
        if groundedness < self.thresholds['groundedness']:
            critical_flags += 1
            failed_metrics.append('groundedness')
        
        if self_contained < self.thresholds['self_contained']:
            critical_flags += 1
            failed_metrics.append('self_contained')
        
        if relevance_emb < self.thresholds['relevance']:
            critical_flags += 1
            failed_metrics.append('relevance')
        
        if consistency < self.thresholds['consistency']:
            critical_flags += 1
            failed_metrics.append('consistency')
        
        if context_precision < self.thresholds['context_precision']:
            critical_flags += 1
            failed_metrics.append('context_precision')
        
        # Status
        if critical_flags == 0:
            status = "EXCELLENT"
        elif critical_flags == 1:
            status = "GOOD"
        elif critical_flags == 2:
            status = "MODERATE"
        else:
            status = "CRITICAL"
        
        print(f"\n{'='*60}")
        print(f"STATUS: {status} ({critical_flags} critical flags)")
        print(f"{'='*60}")
        
        # ==== Aggregazione risultati ====
        return {
            'question_id': question_id,
            'question': question,
            'timestamp': datetime.now().isoformat(),
            
            # Risposte
            'answer_main': answer_main,
            'additional_answers': additional_answers[1:],  # Escludi main (già salvata)
            
            # Chunk recuperati
            'retrieved_chunks': chunks,
            'chunks_metadata': chunks_metadata,
            'num_chunks': len(chunks),
            
            # Metriche
            'metrics': {
                'groundedness': {
                    'score': groundedness,
                    'details': groundedness_details
                },
                'self_contained': {
                    'score': self_contained,
                    'raw_score_1_5': self_contained * 5,
                    'motivation': self_contained_reason
                },
                'relevance': {
                    'score': relevance_emb,
                    'method': 'embedding_similarity'
                },
                'consistency': {
                    'score': consistency,
                    'details': consistency_details
                },
                'context_precision': {
                    'score': context_precision,
                    'chunk_evaluations': chunk_evaluations
                },
                'citation_accuracy': {
                    'score': citation_accuracy,
                    'details': citation_details
                }
            },
            
            # Classificazione
            'critical_flags': critical_flags,
            'failed_metrics': failed_metrics,
            'status': status,
            'thresholds_used': self.thresholds
        }
    
    def validate_dataset(
        self, 
        questions: List[Dict[str, str]],
        test_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Valida un dataset completo di domande
        
        Args:
            questions: Lista di dict con 'question_id' e 'question'
            test_name: Nome test (opzionale, default: auto da timestamp)
        
        Returns:
            Dict con risultati aggregati
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_name = test_name or f"validation_{timestamp}"
        
        print(f"\n{'#'*60}")
        print(f"# INIZIO VALIDAZIONE ZERO-KNOWLEDGE")
        print(f"# Test: {test_name}")
        print(f"# Dataset: {len(questions)} domande")
        print(f"# Consistency runs: {self.num_consistency_runs}")
        print(f"# Timestamp: {timestamp}")
        print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        # Valida ogni domanda
        results = []
        for q in tqdm(questions, desc="Validazione domande"):
            result = self.validate_single_question(
                question=q['question'],
                question_id=q.get('question_id', q.get('user_input'))
            )
            results.append(result)
            
            # Piccola pausa per evitare rate limiting
            time.sleep(0.5)
        
        elapsed_time = time.time() - start_time
        
        # ==== Aggregazione metriche globali ====
        print("\n" + "="*60)
        print("CALCOLO METRICHE AGGREGATE")
        print("="*60)
        
        metrics_aggregate = {
            'groundedness': {
                'mean': sum(r['metrics']['groundedness']['score'] for r in results) / len(results),
                'min': min(r['metrics']['groundedness']['score'] for r in results),
                'max': max(r['metrics']['groundedness']['score'] for r in results),
                'below_threshold': sum(1 for r in results if r['metrics']['groundedness']['score'] < self.thresholds['groundedness'])
            },
            'self_contained': {
                'mean': sum(r['metrics']['self_contained']['score'] for r in results) / len(results),
                'min': min(r['metrics']['self_contained']['score'] for r in results),
                'max': max(r['metrics']['self_contained']['score'] for r in results),
                'below_threshold': sum(1 for r in results if r['metrics']['self_contained']['score'] < self.thresholds['self_contained'])
            },
            'relevance': {
                'mean': sum(r['metrics']['relevance']['score'] for r in results) / len(results),
                'min': min(r['metrics']['relevance']['score'] for r in results),
                'max': max(r['metrics']['relevance']['score'] for r in results),
                'below_threshold': sum(1 for r in results if r['metrics']['relevance']['score'] < self.thresholds['relevance'])
            },
            'consistency': {
                'mean': sum(r['metrics']['consistency']['score'] for r in results) / len(results),
                'min': min(r['metrics']['consistency']['score'] for r in results),
                'max': max(r['metrics']['consistency']['score'] for r in results),
                'below_threshold': sum(1 for r in results if r['metrics']['consistency']['score'] < self.thresholds['consistency'])
            },
            'context_precision': {
                'mean': sum(r['metrics']['context_precision']['score'] for r in results) / len(results),
                'min': min(r['metrics']['context_precision']['score'] for r in results),
                'max': max(r['metrics']['context_precision']['score'] for r in results),
                'below_threshold': sum(1 for r in results if r['metrics']['context_precision']['score'] < self.thresholds['context_precision'])
            },
            'citation_accuracy': {
                'mean': sum(r['metrics']['citation_accuracy']['score'] for r in results) / len(results),
                'min': min(r['metrics']['citation_accuracy']['score'] for r in results),
                'max': max(r['metrics']['citation_accuracy']['score'] for r in results)
            }
        }
        
        # Distribuzione status
        status_counts = {
            'EXCELLENT': sum(1 for r in results if r['status'] == 'EXCELLENT'),
            'GOOD': sum(1 for r in results if r['status'] == 'GOOD'),
            'MODERATE': sum(1 for r in results if r['status'] == 'MODERATE'),
            'CRITICAL': sum(1 for r in results if r['status'] == 'CRITICAL')
        }
        
        # Health Score: % domande OK (EXCELLENT + GOOD)
        health_score = (status_counts['EXCELLENT'] + status_counts['GOOD']) / len(results) * 100
        
        # Domande critiche
        critical_questions = [
            {
                'question_id': r['question_id'],
                'question': r['question'],
                'critical_flags': r['critical_flags'],
                'failed_metrics': r['failed_metrics'],
                'status': r['status']
            }
            for r in results if r['status'] in ['CRITICAL', 'MODERATE']
        ]
        
        # ==== Report finale ====
        final_report = {
            'test_name': test_name,
            'timestamp': timestamp,
            'elapsed_time_seconds': elapsed_time,
            'dataset_size': len(questions),
            'num_consistency_runs': self.num_consistency_runs,
            'thresholds': self.thresholds,
            
            # Metriche aggregate
            'metrics_aggregate': metrics_aggregate,
            
            # Health score
            'health_score': health_score,
            'status_distribution': status_counts,
            'status_percentages': {
                k: (v / len(results) * 100) for k, v in status_counts.items()
            },
            
            # Domande critiche
            'critical_questions': critical_questions,
            'num_critical': len(critical_questions),
            
            # Risultati dettagliati per domanda
            'detailed_results': results
        }
        
        # ==== Salva risultati ====
        self._save_results(final_report, test_name, timestamp)
        
        # ==== Print summary ====
        self._print_summary(final_report)
        
        return final_report
    
    def _save_results(self, report: Dict, test_name: str, timestamp: str):
        """Salva risultati in formato JSON e CSV"""
        
        # JSON completo
        json_path = self.output_dir / f"{test_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Risultati JSON salvati: {json_path}")
        
        # CSV per Excel (solo metriche principali)
        import csv
        csv_path = self.output_dir / f"{test_name}.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'question_id',
                'question',
                'groundedness',
                'self_contained',
                'relevance',
                'consistency',
                'context_precision',
                'citation_accuracy',
                'critical_flags',
                'status',
                'answer_preview',
                'num_chunks'
            ])
            
            # Data
            for r in report['detailed_results']:
                writer.writerow([
                    r['question_id'],
                    r['question'],
                    f"{r['metrics']['groundedness']['score']:.3f}",
                    f"{r['metrics']['self_contained']['score']:.3f}",
                    f"{r['metrics']['relevance']['score']:.3f}",
                    f"{r['metrics']['consistency']['score']:.3f}",
                    f"{r['metrics']['context_precision']['score']:.3f}",
                    f"{r['metrics']['citation_accuracy']['score']:.3f}",
                    r['critical_flags'],
                    r['status'],
                    r['answer_main'][:100] + "...",
                    r['num_chunks']
                ])
        
        print(f"✓ Risultati CSV salvati: {csv_path}")
    
    def _print_summary(self, report: Dict):
        """Stampa summary finale"""
        print(f"\n{'#'*60}")
        print(f"# SUMMARY VALIDAZIONE")
        print(f"{'#'*60}\n")
        
        print(f"Test: {report['test_name']}")
        print(f"Domande: {report['dataset_size']}")
        print(f"Tempo: {report['elapsed_time_seconds']:.1f}s ({report['elapsed_time_seconds']/60:.1f} min)")
        print(f"\n{'='*60}")
        print(f"HEALTH SCORE: {report['health_score']:.1f}%")
        print(f"{'='*60}\n")
        
        print("METRICHE MEDIE:")
        for metric, values in report['metrics_aggregate'].items():
            threshold = report['thresholds'].get(metric, 'N/A')
            below = values.get('below_threshold', 'N/A')
            print(f"  {metric:20s}: {values['mean']:.3f} (threshold: {threshold}, below: {below})")
        
        print(f"\nDISTRIBUZIONE STATUS:")
        for status, count in report['status_distribution'].items():
            pct = report['status_percentages'][status]
            print(f"  {status:12s}: {count:3d} domande ({pct:5.1f}%)")
        
        print(f"\nDOMANDE CRITICHE: {report['num_critical']}")
        if report['critical_questions']:
            print("\nTop 5 domande critiche:")
            for q in report['critical_questions'][:5]:
                print(f"  - {q['question_id']}: {q['question'][:60]}... ({q['critical_flags']} flags)")
        
        print(f"\n{'#'*60}")
        print(f"# VALIDAZIONE COMPLETATA")
        print(f"{'#'*60}\n")

