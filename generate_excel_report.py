import pandas as pd
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# ===========================
# DATI TEST - 7 VALIDAZIONI (LENGTH VERIFIED)
# ===========================

test_data = {
    'Dataset': [
        'ragas_dataset_55q.jsonl',
        'ragas_dataset_55q.jsonl',
        'ragas_dataset_55q.jsonl',
        'ragas_dataset_55q.jsonl',
        'ragas_dataset_55q.jsonl',
        'ragas_dataset_55q.jsonl',
        'ragas_dataset_44q.jsonl'
    ],
    'N° Test': [1, 2, 3, 4, 5, 6, 7],
    'Data Test': [
        '2026-02-16 17:01',
        '2026-02-16 17:04',
        '2026-02-16 17:41',
        '2026-02-17 11:26',
        '2026-02-17 11:26',
        '2026-02-17 21:49',
        '2026-02-17 21:49'
    ],
    'Configurazioni Chatbot': [
        'Hybrid BM25+FAISS\nWeights: 60/40 static\nTop-K: 10\nReranking: NO\nQuery Expansion: NO',
        
        'Hybrid BM25+FAISS\nWeights: Dynamic by intent\nTop-K: 10\nIntent Strategies: YES\n- LLM Direct\n- Procedural\n- Evidence',
        
        'Hybrid BM25+FAISS\nWeights: Dynamic\nTop-K: 15\nIntent Strategies: YES\nCanonical Tags: ACTIVE\nAcronym Expansion: ACTIVE',
        
        'Hybrid BM25+FAISS\nWeights: Dynamic tuned\nTop-K: 10\nIntent tuning:\n- Acronym: 30/70 BM25\n- Definitional: 40/60\n- Procedural: 70/30',
        
        'Hybrid BM25+FAISS\nCrossEncoder Reranking: YES\nModel: ms-marco-MiniLM-L12\nFetch: 30 → Rerank: 10\nDeduplication: ACTIVE',
        
        'Hybrid BM25+FAISS\nQuery Expansion: YES\n- Synonyms (max 5 queries)\n- Intent enrichment\nReranking: YES (30→10)\nEnsemble weights tuned',
        
        'Same as T006\nDataset CLEANED (44q):\n- No EMAIL_MISMATCH\n- No missing info\n- No OOD questions'
    ],
    'Faithfulness': [0.747, 0.747, 0.807, 0.786, 0.807, 0.839, 0.858],
    'Context Recall': [0.687, 0.687, 0.667, 0.650, 0.687, 0.603, 0.648],
    'Context Precision': ['N/A', 'N/A', 0.604, 0.608, 0.742, 0.682, 0.732],
    'Answer Relevancy': [0.655, 0.655, 0.627, 0.643, 0.655, 0.641, 0.664],
    'Answer Correctness': [0.494, 0.494, 0.461, 0.461, 0.494, 0.454, 0.492],
    'Health Score': [14.9, 14.9, 0.0, 0.0, 0.0, 0.0, 56.1],
    'Note Analitiche': [
        'T001 BASELINE: Intent-based multi-path strategy (LLM Direct, Procedural, Evidence). Faithfulness -4.9% vs baseline. Context recall +1.8pp but relevancy -3.4pp. ISSUE: Intent classification too simple, 10 outliers faith=0. Health 14.9 (Poor). DECISION: Abandon multi-path.',
        
        'T002 CANONICAL TAGS: Faithfulness recovers (+6%). Acronym expansion active. TRADE-OFF: Relevancy -6.2%, correctness -3.1%. Context precision 60.4% (chunk quality sub-optimal). ROOT CAUSE: Multi-path confuses LLM. Health 0. NEXT: Test reranking.',
            
        'T003 k=10 TUNING: Reduced top-k 15→10. Intent weights tuned. NEGATIVE: Recall drops 65.0%, faithfulness below baseline. INSIGHT: k=10 too low for multi-hop. POSITIVE: Precision +0.4pp. Health 0. DECISION: Reranking with k=30.',
            
        'T004 RERANKING BREAKTHROUGH: CrossEncoder ms-marco. Fetch 30→rerank 10. POSITIVE: Faithfulness 80.7% (+2.1pp), precision +13.4pp (74.2%), recall recovers 68.7%. Correlations improved. PROBLEM: Correctness still 49.4%. Health Poor but base metrics solid. NEXT: Query expansion.',
            
        'T005 QUERY EXPANSION: Multi-query with synonyms + intent (max 5). MIXED: Faithfulness PEAK 83.9% BUT recall DROPS 60.3% (-8.4pp), correctness worsens 45.4%. ROOT CAUSE: Budget k=30/5 queries = 6 docs/query TOO LOW. 11 EMAIL_MISMATCH questions distort metrics. Health 0. ACTION: Recalculate cleaned.',
            
        'T006 RAW RESULTS: Same config as T005 but with full 55 questions dataset. Faithfulness 83.9% (PEAK across all tests). Context precision 68.2%. Answer correctness 45.4% (lowest). 11 questions with EMAIL_MISMATCH or missing info cause metric distortion. Health 0 due to outlier penalties. DECISION: Analyze with cleaned dataset.',
            
        'T007 CLEANED DATASET: Excluded 11 impossible questions (EMAIL_MISMATCH, missing info, OOD). 44 valid questions. RESULTS: Faithfulness 85.8% (+6.2%) EXCELLENT, 75% questions ≥0.8, min=0.167. Precision 73.2%. Correctness 49.2% (same as baseline - LLM issue not retrieval). Recall 64.8% (reranking trade-off). CRITICAL: 3/44 (6.8%). Health 56.1 (MODERATE) +40.4 points! SYSTEM READY FOR TUNING. NEXT: Glossary injection, answer extraction, increase K 30→45.'
    ]
}

# Verify lengths before creating DataFrame
print("Verifying data lengths:")
for key, value in test_data.items():
    print(f"  {key}: {len(value)}")

df_results = pd.DataFrame(test_data)

# ===========================
# METRICS EVOLUTION
# ===========================

metrics_evolution = {
    'Metrica': ['Faithfulness', 'Context Recall', 'Context Precision', 'Answer Relevancy', 'Answer Correctness', 'Health Score'],
    'T001': [0.747, 0.687, 'N/A', 0.655, 0.494, 14.9],
    'T002': [0.807, 0.667, 0.604, 0.627, 0.461, 0.0],
    'T003': [0.786, 0.650, 0.608, 0.643, 0.461, 0.0],
    'T004': [0.807, 0.687, 0.742, 0.655, 0.494, 0.0],
    'T005': [0.839, 0.603, 0.682, 0.641, 0.454, 0.0],
    'T006': [0.858, 0.648, 0.732, 0.664, 0.492, 56.1],
    'Delta': ['+0.111', '-0.039', '+0.128', '+0.009', '-0.002', '+41.2'],
    'Trend': ['↑↑', '→', '↑↑', '→', '→', '↑↑']
}

df_evolution = pd.DataFrame(metrics_evolution)

# ===========================
# CONFIGURATION HISTORY
# ===========================

config_history = {
    'Test': ['T001', 'T002', 'T003', 'T004', 'T005', 'T006', 'T006b'],
    'Data': ['02-16', '02-16', '02-16', '02-17', '02-17', '02-17', '02-17'],
    'Top-K': [10, 10, 15, 10, 30, 30, 30],
    'Reranking': ['NO', 'NO', 'NO', 'NO', 'YES', 'YES', 'YES'],
    'Query Exp': ['NO', 'NO', 'Canon', 'Canon', 'Canon', 'YES', 'YES'],
    'Intent Strat': ['YES', 'YES', 'YES', 'YES', 'NO', 'NO', 'NO'],
    'Dataset': [55, 55, 55, 55, 55, 55, 44],
    'Key Changes': [
        'Multi-path strategies setup',
        'Canonical tags + acronym expansion',
        'Reduced k to 10',
        'CrossEncoder reranking',
        'Query expansion (5 queries)',
        'Dataset cleaned',
        'Final config'
    ]
}

df_config = pd.DataFrame(config_history)

# ===========================
# CRITICAL QUESTIONS
# ===========================

critical_questions = {
    'Test': ['T001', 'T001', 'T001', 'T002', 'T004', 'T005', 'T006', 'T006', 'T006'],
    'Question': ['Q010', 'Q014', 'Q025', 'Q010', 'Q038', 'Q029', 'Q010', 'Q014', 'Q034'],
    'Issue': [
        'OdC acronimo?',
        'Q1-Q4 cruscotti?',
        'Partita IVA?',
        'OdC acronimo?',
        'Vet incaricato vs aziendale?',
        'Password non ricevuta?',
        'OdC acronimo?',
        'Q1-Q4 cruscotti?',
        'Non trovo allevamento?'
    ],
    'Metrics': [
        'F=0.67 AR=0.0',
        'F=1.0 AR=0.0',
        'F=0.0 AR=0.0',
        'F=0.67 AR=0.0',
        'F=0.14 R=0.0',
        'F=0.17 AR=0.0',
        'F=0.33 AR=0.0',
        'F=1.0 AR=0.0',
        'F=1.0 AR=0.37'
    ],
    'Status': [
        'Persistent',
        'Persistent',
        'EXCLUDED',
        'Persistent',
        'New',
        'EXCLUDED',
        'CRITICAL',
        'CRITICAL',
        'CRITICAL'
    ],
    'Fix': [
        'Glossary OdC',
        'Glossary Q1-Q4',
        'Fallback "Non so"',
        'Answer extraction',
        'Multi-doc synthesis',
        'KB expansion',
        'P0: Glossary',
        'P0: Tech dictionary',
        'Procedural orchestrator'
    ]
}

df_critical = pd.DataFrame(critical_questions)

# ===========================
# RECOMMENDATIONS
# ===========================

recommendations = {
    'Priority': ['P0', 'P0', 'P0', 'P1', 'P1', 'P1', 'P2', 'P2', 'P3'],
    'Action': [
        'Glossary injection (OdC, Q1-Q4, BDN)',
        'Implement "Non so" response',
        'Email whitelist fix',
        'Increase K: 30→45',
        'Answer extraction post-rerank',
        'Few-shot prompting',
        'Multi-doc synthesis',
        'Procedural orchestrator',
        'Chain-of-thought'
    ],
    'Impact': [
        '+10pp Health',
        '+10pp Health',
        'Resolve 4 questions',
        '+5pp Recall',
        '+3pp Correctness',
        '+5pp Relevancy',
        '+5pp Recall',
        '+10pp Correctness',
        '+15pp Correctness'
    ],
    'Effort': ['2d', '1d', '1d', '1h', '3d', '2d', '7d', '14d', '60d'],
    'Status': ['🔴', '🔴', '🔴', '🔴', '🔴', '🔴', '🟡', '🟡', '⚪']
}

df_recommendations = pd.DataFrame(recommendations)

# ===========================
# CREATE EXCEL FILE
# ===========================

file_path = 'RAG_Validation_Progress_Report.xlsx'

with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='Summary', index=False)
    df_evolution.to_excel(writer, sheet_name='Metrics', index=False)
    df_config.to_excel(writer, sheet_name='Config', index=False)
    df_critical.to_excel(writer, sheet_name='Critical', index=False)
    df_recommendations.to_excel(writer, sheet_name='Roadmap', index=False)

# ===========================
# FORMATTING
# ===========================

wb = openpyxl.load_workbook(file_path)

# Common styles
header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF", size=11)

# Format all sheets headers
for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

# Sheet 1: Summary
ws1 = wb['Summary']
ws1.column_dimensions['A'].width = 25
ws1.column_dimensions['D'].width = 40
ws1.column_dimensions['K'].width = 70

for row in ws1.iter_rows(min_row=2, max_row=ws1.max_row):
    ws1.row_dimensions[row[0].row].height = 90
    for cell in row:
        cell.alignment = Alignment(vertical='top', wrap_text=True)

# Health Score colors
for row in range(2, ws1.max_row + 1):
    cell = ws1[f'J{row}']
    if cell.value and isinstance(cell.value, (int, float)):
        score = float(cell.value)
        if score < 20:
            cell.fill = PatternFill(start_color="FFC7CE", fill_type="solid")
        elif score < 60:
            cell.fill = PatternFill(start_color="FFEB9C", fill_type="solid")
        else:
            cell.fill = PatternFill(start_color="C6EFCE", fill_type="solid")

# Sheet 2: Metrics
ws2 = wb['Metrics']
for row in range(2, ws2.max_row + 1):
    trend = ws2[f'H{row}'].value
    if trend and '↑' in str(trend):
        ws2[f'H{row}'].font = Font(color="00B050", bold=True, size=14)
    elif trend and '→' in str(trend):
        ws2[f'H{row}'].font = Font(color="FFA500", bold=True, size=14)

# Sheet 4: Critical
ws4 = wb['Critical']
ws4.column_dimensions['C'].width = 35
ws4.column_dimensions['F'].width = 35

for row in range(2, ws4.max_row + 1):
    status = ws4[f'E{row}'].value
    if status == 'CRITICAL':
        ws4[f'E{row}'].fill = PatternFill(start_color="C00000", fill_type="solid")
        ws4[f'E{row}'].font = Font(bold=True, color="FFFFFF")
    elif status == 'EXCLUDED':
        ws4[f'E{row}'].fill = PatternFill(start_color="D3D3D3", fill_type="solid")

# Sheet 5: Roadmap
ws5 = wb['Roadmap']
ws5.column_dimensions['B'].width = 40

priority_colors = {'P0': "C00000", 'P1': "FFA500", 'P2': "92D050", 'P3': "00B0F0"}
for row in range(2, ws5.max_row + 1):
    priority = ws5[f'A{row}'].value
    if priority in priority_colors:
        ws5[f'A{row}'].fill = PatternFill(start_color=priority_colors[priority], fill_type="solid")
        ws5[f'A{row}'].font = Font(bold=True, color="FFFFFF")

wb.save(file_path)

print(f"\n{'='*60}")
print(f"✅ FILE EXCEL CREATO!")
print(f"{'='*60}")
print(f"📂 File: {file_path}")
print(f"📊 5 Sheets: Summary, Metrics, Config, Critical, Roadmap")
print(f"\n📈 Key Results:")
print(f"   Faithfulness: 0.747 → 0.858 (+14.9%)")
print(f"   Health Score: 14.9 → 56.1 (+276%)")
print(f"   Critical Issues: 3/44 (6.8%)")
print(f"\n🎯 Next: P0 fixes → Health 70-75 (GOOD)")
print(f"{'='*60}\n")