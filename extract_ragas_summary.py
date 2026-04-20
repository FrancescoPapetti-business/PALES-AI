"""
extract_ragas_summary.py

Estrae solo domande, risposte, ground truth e metriche dal report CSV creato da ragas_eval.py,
creando un file di output più leggibile e compatto.

Requirements:
  pip install pandas openpyxl

Usage:
  python extract_ragas_summary.py --input results/\ragas_report_20260217_123143.csv --output results/ragas_summary.csv
  
  # Per output Excel:
  python extract_ragas_summary.py --input results/ragas_report_20260211_143045.csv --output results/ragas_summary.xlsx
"""

import argparse
import pandas as pd
from pathlib import Path


def extract_summary(input_path: Path, output_path: Path):
    """
    Estrae le colonne rilevanti dal report RAGAS completo.
    
    Args:
        input_path: Path al file CSV generato da ragas_eval.py
        output_path: Path dove salvare il file estratto
    """
    
    print(f"📖 Caricamento report da: {input_path}")
    
    # Carica il CSV completo
    df = pd.read_csv(input_path)
    
    print(f"   Trovate {len(df)} righe nel report originale")
    
    # Colonne da estrarre (nomi corretti dal report RAGAS)
    columns_to_extract = [
        'user_input',           # Testo della domanda
        'response',             # Risposta generata dal chatbot
        'reference',            # Ground truth (risposta corretta di riferimento)
        'faithfulness',         # Fedeltà della risposta al contesto
        'answer_relevancy',     # Rilevanza della risposta
        'context_precision',    # Precisione del contesto recuperato
        'context_recall',       # Completezza del contesto recuperato
        'answer_correctness',   # Correttezza semantica della risposta
        'factual_mismatch',     # Flag di mismatch fattuale
        'answerable'            # Flag se la domanda era answerable
    ]
    
    # Verifica che le colonne esistano
    missing_cols = [col for col in columns_to_extract if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Attenzione: colonne non trovate nel report: {missing_cols}")
        print(f"   Colonne disponibili: {list(df.columns)}")
        # Prendi solo le colonne che esistono
        columns_to_extract = [col for col in columns_to_extract if col in df.columns]
    
    # Estrai solo le colonne desiderate
    df_summary = df[columns_to_extract].copy()
    
    # Opzionale: arrotonda le metriche per maggiore leggibilità
    metric_columns = ['faithfulness', 'answer_relevancy', 'context_precision', 
                      'context_recall', 'answer_correctness']
    for col in metric_columns:
        if col in df_summary.columns:
            df_summary[col] = df_summary[col].round(4)
    
    # Salva il risultato
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.xlsx':
        df_summary.to_excel(output_path, index=False, engine='openpyxl')
    else:
        df_summary.to_csv(output_path, index=False)
    
    print(f"✅ Summary salvato in: {output_path}")
    print(f"   Colonne estratte: {list(df_summary.columns)}")
    
    # Stampa anche le metriche aggregate
    print("\n📊 Metriche aggregate (media):")
    for col in metric_columns:
        if col in df_summary.columns:
            mean_val = df_summary[col].mean()
            print(f"   {col:25s}: {mean_val:.4f}")
    
    # Statistiche sui flag
    if 'factual_mismatch' in df_summary.columns:
        factual_count = df_summary['factual_mismatch'].sum()
        print(f"\n🔍 Factual mismatches: {factual_count}/{len(df_summary)}")
    
    if 'answerable' in df_summary.columns:
        answerable_count = df_summary['answerable'].sum()
        print(f"✅ Domande answerable: {answerable_count}/{len(df_summary)}")


def main():
    parser = argparse.ArgumentParser(
        description="Estrae domande, risposte, ground truth e metriche dal report RAGAS"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path al file CSV generato da ragas_eval.py"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path dove salvare il summary (.csv o .xlsx)"
    )
    
    args = parser.parse_args()
    
    # Verifica che il file di input esista
    if not args.input.exists():
        print(f"❌ ERROR: File di input non trovato: {args.input}")
        return
    
    # Estrai il summary
    extract_summary(args.input, args.output)


if __name__ == "__main__":
    main()