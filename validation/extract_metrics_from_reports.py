"""
extract_metrics_from_reports.py

Estrae metriche da report RAGAS CSV esistenti e genera un'analisi statistica avanzata.

Funzionalità:
- Legge report CSV con metriche RAGAS
- Mappa question_id da gold_dataset.jsonl
- Calcola statistiche descrittive complete per ogni metrica
- Identifica outliers e domande critiche
- Genera health score del sistema
- Calcola correlazioni tra metriche
- Output in formato JSONL con summary statistico

Usage:
  # Singolo report
  python validation/extract_metrics_from_reports.py --report results/\ragas_report_20260304_103447.csv
  
  # Multi-report con wildcard
  python validation/extract_metrics_from_reports.py --report "results/ragas_report_*.csv"
  
  # Con path gold custom
  python validation/extract_metrics_from_reports.py --report results/ragas_report_20260203.csv --gold validation/gold_dataset.jsonl

Requirements:
  pip install pandas numpy
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import glob

# Costanti
METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy", 
    "context_precision",
    "context_recall",
    "answer_correctness"
]


def load_gold_dataset(gold_path: Path) -> Dict[str, str]:
    """
    Carica gold_dataset.jsonl e crea mapping question -> question_id.
    
    Args:
        gold_path: Path al file JSONL gold dataset
    
    Returns:
        Dict con chiave=question (lowercase, stripped), valore=question_id
        
    Example:
        >>> mapping = load_gold_dataset(Path("validation/gold_dataset.jsonl"))
        >>> mapping["come accedo a classyfarm?"]
        'gold001'
    """
    mapping = {}
    
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                question = data.get("question", "").strip().lower()
                qid = data.get("metadata", {}).get("question_id", "unknown")
                
                if question:
                    mapping[question] = qid
            except json.JSONDecodeError:
                continue
    
    return mapping


def compute_statistics(series: pd.Series) -> Dict[str, Any]:
    """
    Calcola statistiche descrittive complete per una Serie Pandas.
    
    Calcola:
    - Statistiche centrali: mean, median, mode
    - Statistiche di dispersione: std, variance, range, quartili, IQR
    - Outliers detection con metodo IQR (1.5 * IQR)
    
    Args:
        series: Serie Pandas con valori numerici (metriche)
    
    Returns:
        Dict con tutte le statistiche calcolate
        
    Example:
        >>> import pandas as pd
        >>> s = pd.Series([0.8, 0.85, 0.9, 0.92, 0.95])
        >>> stats = compute_statistics(s)
        >>> stats['mean']
        0.884
    """
    # Rimuovi NaN per calcoli
    clean_series = series.dropna()
    
    if len(clean_series) == 0:
        return {
            "mean": None,
            "median": None,
            "mode": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "outliers_low_count": 0,
            "outliers_high_count": 0
        }
    
    # Statistiche centrali
    q1 = clean_series.quantile(0.25)
    q2 = clean_series.quantile(0.50)  # mediana
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1
    
    # Outliers detection (metodo IQR standard)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_low = clean_series[clean_series < lower_bound]
    outliers_high = clean_series[clean_series > upper_bound]
    
    # Mode handling (può esserci multipli o nessuno)
    mode_values = clean_series.mode()
    mode = round(mode_values.iloc[0], 4) if len(mode_values) > 0 else None
    
    return {
        "mean": round(clean_series.mean(), 4),
        "median": round(q2, 4),
        "mode": mode,
        "std": round(clean_series.std(), 4),
        "variance": round(clean_series.var(), 4),
        "min": round(clean_series.min(), 4),
        "max": round(clean_series.max(), 4),
        "range": round(clean_series.max() - clean_series.min(), 4),
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "outliers_low_count": len(outliers_low),
        "outliers_high_count": len(outliers_high)
    }


def identify_critical_questions(df: pd.DataFrame, stats: Dict) -> List[Dict]:
    """
    Identifica domande problematiche basandosi su outliers bassi.
    
    Una domanda è critica se ha almeno una metrica con valore outlier basso
    (indica performance sotto standard).
    
    Args:
        df: DataFrame con question_id, question e metriche
        stats: Dict con statistiche per ogni metrica (da compute_statistics)
    
    Returns:
        Lista di dict ordinata per severità, ogni dict contiene:
        - question_id: ID della domanda
        - question: Testo della domanda
        - reason: Descrizione degli outliers
        - affected_metrics: Lista metriche problematiche
        - severity: "high" se 2+ metriche, "medium" se 1 metrica
        
    Example:
        >>> critical = identify_critical_questions(df, stats)
        >>> critical[0]['severity']
        'high'
    """
    critical = []
    
    for idx, row in df.iterrows():
        qid = row.get("question_id", "unknown")
        question = row.get("question", "")
        affected = []
        reasons = []
        
        for metric in METRIC_COLUMNS:
            value = row[metric]
            
            # Skip NaN values
            if pd.isna(value):
                continue
            
            metric_stats = stats[metric]
            
            # Check se outlier basso (problematico)
            if metric_stats["q1"] is not None and metric_stats["iqr"] is not None:
                threshold_low = metric_stats["q1"] - 1.5 * metric_stats["iqr"]
                
                if value < threshold_low:
                    affected.append(metric)
                    reasons.append(f"Low {metric} ({value:.2f})")
        
        if affected:
            critical.append({
                "question_id": qid,
                "question": question,
                "reason": "; ".join(reasons),
                "affected_metrics": affected,
                "severity": "high" if len(affected) >= 2 else "medium"
            })
    
    # Ordina per severità: prima high, poi per numero metriche affette
    return sorted(
        critical, 
        key=lambda x: (0 if x["severity"] == "high" else 1, -len(x["affected_metrics"]))
    )


def compute_correlations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calcola correlazione Pearson tra tutte le coppie di metriche.
    
    Restituisce solo la parte triangolare superiore della matrice di correlazione
    per evitare duplicati (corr(A,B) = corr(B,A)).
    
    Args:
        df: DataFrame con colonne delle metriche RAGAS
    
    Returns:
        Dict con chiave "metricA_vs_metricB" e valore coefficiente Pearson
        
    Example:
        >>> corr = compute_correlations(df)
        >>> corr['faithfulness_vs_answer_relevancy']
        0.8234
    """
    corr_matrix = df[METRIC_COLUMNS].corr(method='pearson')
    correlations = {}
    
    # Solo triangolo superiore (evita duplicati e diagonale)
    for i, metric1 in enumerate(METRIC_COLUMNS):
        for metric2 in METRIC_COLUMNS[i+1:]:
            key = f"{metric1}_vs_{metric2}"
            value = corr_matrix.loc[metric1, metric2]
            
            # Gestisci NaN (può accadere se tutte le metriche sono costanti)
            correlations[key] = round(value, 4) if pd.notna(value) else None
    
    return correlations


def calculate_health_score(stats: Dict, critical: List) -> Dict:
    """
    Calcola uno score di salute generale del sistema (0-100).
    
    Formula:
    - Base score (0-60): Media di tutte le metriche * 60
    - Penalità outliers bassi (max -20): Numero totale outliers bassi * 2
    - Penalità domande critiche (max -20): Numero domande critiche * 3
    - Score finale: max(0, base_score - penalità)
    
    Rating:
    - excellent: ≥ 90
    - good: ≥ 75
    - fair: ≥ 60
    - poor: < 60
    
    Args:
        stats: Dict con statistiche per ogni metrica
        critical: Lista domande critiche da identify_critical_questions
    
    Returns:
        Dict con overall_score, rating e breakdown dettagliato
        
    Example:
        >>> health = calculate_health_score(stats, critical_questions)
        >>> health['rating']
        'good'
    """
    # Score basato su medie (0-60 punti)
    avg_metrics = []
    for metric in METRIC_COLUMNS:
        mean_val = stats[metric].get("mean")
        if mean_val is not None:
            avg_metrics.append(mean_val)
    
    if not avg_metrics:
        base_score = 0.0
    else:
        base_score = (sum(avg_metrics) / len(avg_metrics)) * 60
    
    # Penalità per outliers bassi (max -20 punti)
    total_outliers_low = sum(
        stats[m].get("outliers_low_count", 0) for m in METRIC_COLUMNS
    )
    outlier_penalty = min(total_outliers_low * 2, 20)
    
    # Penalità per domande critiche (max -20 punti)
    critical_penalty = min(len(critical) * 3, 20)
    
    # Score finale
    final_score = max(0, base_score - outlier_penalty - critical_penalty)
    
    # Determina rating
    if final_score >= 90:
        rating = "excellent"
    elif final_score >= 75:
        rating = "good"
    elif final_score >= 60:
        rating = "fair"
    else:
        rating = "poor"
    
    return {
        "overall_score": round(final_score, 2),
        "rating": rating,
        "breakdown": {
            "base_score": round(base_score, 2),
            "outlier_penalty": outlier_penalty,
            "critical_penalty": critical_penalty
        }
    }


def process_report(report_path: Path, gold_mapping: Dict[str, str]) -> Dict:
    """
    Processa un singolo report CSV.
    
    Workflow:
    1. Legge CSV e valida colonne
    2. Mappa question_id usando gold_mapping
    3. Calcola statistiche per ogni metrica
    4. Identifica outliers per metrica
    5. Calcola correlazioni tra metriche
    6. Identifica domande critiche
    7. Calcola health score
    
    Args:
        report_path: Path al file CSV report RAGAS
        gold_mapping: Dict question → question_id da load_gold_dataset
    
    Returns:
        Dict con due chiavi:
        - "records": Lista di dict per ogni domanda (per JSONL)
        - "summary": Dict con statistical_summary completo
        
    Raises:
        ValueError: Se mancano colonne richieste nel CSV
        
    Example:
        >>> result = process_report(Path("results/ragas_report.csv"), mapping)
        >>> len(result['records'])
        10
    """
    print(f"\n📊 Processing: {report_path.name}")
    
    # 1. Leggi CSV
    try:
        df = pd.read_csv(report_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")
    
    # Verifica colonne richieste
    required_cols = ["user_input"] + METRIC_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    
    # 2. Mappa question_id
    def map_question_id(question_text):
        key = question_text.strip().lower()
        return gold_mapping.get(key, "unknown")
    
    df["question_id"] = df["user_input"].apply(map_question_id)
    
    # Rinomina per output
    df = df.rename(columns={"user_input": "question"})
    
    # 3. Calcola statistiche per metrica
    stats = {}
    for metric in METRIC_COLUMNS:
        stats[metric] = compute_statistics(df[metric])
        
        # Aggiungi lista outliers (question_id)
        q1 = stats[metric]["q1"]
        q3 = stats[metric]["q3"]
        iqr = stats[metric]["iqr"]
        
        if q1 is not None and iqr is not None:
            outliers_low_ids = df[
                df[metric] < (q1 - 1.5 * iqr)
            ]["question_id"].tolist()
            
            outliers_high_ids = df[
                df[metric] > (q3 + 1.5 * iqr)
            ]["question_id"].tolist()
        else:
            outliers_low_ids = []
            outliers_high_ids = []
        
        stats[metric]["outliers_low"] = outliers_low_ids
        stats[metric]["outliers_high"] = outliers_high_ids
    
    # 4. Correlazioni
    correlations = compute_correlations(df)
    
    # 5. Domande critiche
    critical = identify_critical_questions(df, stats)
    
    # 6. Health score
    health = calculate_health_score(stats, critical)
    
    # 7. Costruisci output
    results = []
    
    # Aggiungi ogni domanda
    for _, row in df.iterrows():
        record = {
            "question_id": row["question_id"],
            "question": row["question"]
        }
        for metric in METRIC_COLUMNS:
            value = row[metric]
            record[metric] = round(value, 4) if pd.notna(value) else None
        results.append(record)
    
    # Aggiungi summary statistico
    summary = {
        "statistical_summary": {
            "total_questions": len(df),
            "source_report": report_path.name,
            "timestamp": datetime.now().isoformat(),
            **{metric: stats[metric] for metric in METRIC_COLUMNS},
            "correlations": correlations,
            "critical_questions": critical,
            "health_score": health
        }
    }
    
    return {"records": results, "summary": summary}


def main():
    """Entry point principale per lo script."""
    parser = argparse.ArgumentParser(
        description="Extract and analyze metrics from RAGAS CSV reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Singolo report
  python validation/extract_metrics_from_reports.py --report results/ragas_report_20260203.csv
  
  # Multi-report con wildcard
  python validation/extract_metrics_from_reports.py --report "results/ragas_report_*.csv"
  
  # Output in directory custom
  python validation/extract_metrics_from_reports.py --report results/ragas_report.csv --out-dir analysis
        """
    )
    
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to CSV report (supports wildcards: results/ragas_*.csv)"
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("validation/gold_dataset.jsonl"),
        help="Path to gold dataset JSONL (default: validation/gold_dataset.jsonl)"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for analysis JSONL (default: results)"
    )
    
    args = parser.parse_args()
    
    # Valida gold dataset
    if not args.gold.exists():
        print(f"❌ Gold dataset not found: {args.gold}")
        print(f"   Expected location: {args.gold.absolute()}")
        print(f"   Make sure the file exists or specify --gold path")
        return 1
    
    # Carica mapping
    print(f"📂 Loading gold dataset: {args.gold}")
    try:
        gold_mapping = load_gold_dataset(args.gold)
        print(f"   ✅ Loaded {len(gold_mapping)} question mappings")
    except Exception as e:
        print(f"❌ Error loading gold dataset: {e}")
        return 1
    
    # Trova report files (supporta wildcards)
    report_files = sorted(glob.glob(args.report))
    if not report_files:
        print(f"❌ No report files found matching: {args.report}")
        return 1
    
    print(f"\n🔍 Found {len(report_files)} report(s) to process")
    
    # Processa ogni report
    success_count = 0
    for report_path in report_files:
        report_path = Path(report_path)
        
        try:
            result = process_report(report_path, gold_mapping)
            
            # Genera output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"ragas_metrics_analysis_{report_path.stem}_{timestamp}.jsonl"
            output_path = args.out_dir / output_name
            
            # Scrivi JSONL
            args.out_dir.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                # Scrivi records (una riga per record)
                for record in result["records"]:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
                # Scrivi summary (pretty-print)
                f.write(json.dumps(result["summary"], ensure_ascii=False, indent=2))
            
            print(f"   ✅ Analysis saved: {output_path}")
            
            # Print summary to console
            summary = result["summary"]["statistical_summary"]
            health = summary["health_score"]
            
            print(f"\n   📈 REPORT SUMMARY")
            print(f"      Questions: {summary['total_questions']}")
            print(f"      Health Score: {health['overall_score']}/100 ({health['rating'].upper()})")
            print(f"      Critical Questions: {len(summary['critical_questions'])}")
            
            if summary['critical_questions']:
                print(f"\n   ⚠️  TOP CRITICAL QUESTIONS:")
                for cq in summary['critical_questions'][:3]:
                    print(f"      • {cq['question_id']}: {cq['reason']}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {report_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary finale
    print(f"\n{'='*60}")
    print(f"✅ Successfully processed {success_count}/{len(report_files)} reports")
    print(f"📂 Output directory: {args.out_dir.absolute()}")
    
    return 0 if success_count == len(report_files) else 1


if __name__ == "__main__":
    exit(main())