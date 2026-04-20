"""
excel_to_jsonl.py

Helper script to convert a curated Excel/CSV file into the JSONL format
required for RAGAS evaluation.

Input columns expected in the Excel/CSV:
  - question (required)
  - ground_truth (required)
  - role (optional, default: operatore)

Usage:
  python tools/validation/excel_to_jsonl.py --input my_qa_pairs.xlsx --output validation/gold_dataset.jsonl
"""

import argparse
import pandas as pd
import json
import sys
from pathlib import Path

def main():
    # 0. Resolve Project Root
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # validation/ -> root

    parser = argparse.ArgumentParser(description="Convert Excel/CSV to RAGAS JSONL")
    parser.add_argument("--input", type=Path, default=Path("qa_dataset.xlsx"), help="Path to input .xlsx or .csv file")
    parser.add_argument("--output", type=Path, default=Path("validation/gold_dataset.jsonl"), help="Path to output .jsonl file")
    args = parser.parse_args()

    # 1. Smart Path Resolution (Check CWD then Root)
    input_path = args.input
    if not input_path.exists():
        # Try looking in project root
        input_path = project_root / args.input.name

    if not input_path.exists():
        print(f"❌ Input file not found: {args.input}")
        print(f"   Cercato anche in: {project_root}")
        sys.exit(1)

    # 2. Load Data
    print(f"📂 Reading {input_path}...")
    if input_path.suffix.lower() == '.csv':
        df = pd.read_csv(input_path)
    else:
        df = pd.read_excel(input_path)

    # 3. Validate Columns
    required_cols = ['question', 'ground_truth']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing required column: '{col}'")
            print(f"   Found columns: {list(df.columns)}")
            sys.exit(1)

    # 3. Convert to JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # Capture extra columns for metadata (e.g. 'source_doc', 'page')
            meta = {}
            for col in df.columns:
                if col not in ['question', 'ground_truth', 'role']:
                    if pd.notna(row[col]):
                        meta[col] = str(row[col]).strip()

            record = {
                "question": str(row['question']).strip(),
                "ground_truth": str(row['ground_truth']).strip(),
                "role": str(row.get('role', 'operatore')).strip(),
                "metadata": meta
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"✅ Converted {count} rows to {args.output}")

if __name__ == "__main__":
    main()