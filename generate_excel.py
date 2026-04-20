import pandas as pd
from datetime import datetime
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

# ===========================
# DATI TEST
# ===========================

test_data = {
    'Dataset': [
        'gold_dataset_25.jsonl',
        'gold_dataset_25.jsonl'
    ],
    'N° Test': [1, 2],
    'Data Test': [
        '2026-02-06 10:27:39',
        '2026-02-07 16:44:31'
    ],
    'Configurazioni Chatbot': [
        'BM25: Non funzionante (indice vuoto)\nFAISS: text-embedding-3-small\nEnsemble: Solo FAISS\nTop-K: 5\nCanonical Concepts: Attivi',
        'BM25: Rigenerato + preprocessing\nFAISS: text-embedding-3-small\nEnsemble: BM25+FAISS (weights [0.5,0.5])\nTop-K: 10\nCanonical Concepts: Attivi\nPreprocessing: Rimozione apostrofi'
    ],
    'Faithfulness (μ)': [0.8307, 0.8674],
    'Context Recall (μ)': [0.6300, 0.7033],
    'Context Precision (μ)': [0.5608, 0.5972],
    'Answer Relevancy (μ)': [0.5921, 0.6704],
    'Answer Correctness (μ)': [0.4660, 0.5185],
    'Health Score': [21.96, 21.28],
    'Note Analitiche': [
        'BASELINE TEST. BM25 non funzionante (indice vuoto). Retrieval solo FAISS. '
        'Performance mediocri: 8 query con context_precision=0 (no retrieval). '
        'Domande critiche: CF_GOLD_102 (delegato faithfulness=0.33), CF_GOLD_116 (faithfulness=0), CF_GOLD_123 (BDN faithfulness=0.33). '
        'Top-K=5 troppo basso: chunk rank 9+ persi. '
        'Necessari fix urgenti: rigenerazione BM25 + preprocessing query + aumento top-k.',
        
        'POST-FIX BM25. Context recall +11.6% (chunk rank 9 ora recuperato). '
        'Risolti: CF_GOLD_102 recall 0→1, CF_GOLD_116 faithfulness 0→0.5, CF_GOLD_123 faithfulness 0.33→1.0. '
        'Trade-off: outlier penalty aumentato (-10 vs -6), health score -3% nonostante base score +9%. '
        'BM25 rank ancora sub-ottimale (rank 18 per "delegato"). '
        'Prossimi step: rimozione canonical concepts + upgrade embeddings FAISS-large.'
    ]
}

# Creazione DataFrame
df_results = pd.DataFrame(test_data)

# ===========================
# DATI CONFIGURAZIONI DETTAGLIATE
# ===========================

config_history = {
    'Test ID': [1, 2],
    'Data': ['2026-02-06', '2026-02-07'],
    'BM25 Status': ['Non funzionante', 'Attivo + Preprocessato'],
    'BM25 Preprocessing': ['N/A', 'Rimozione apostrofi, punteggiatura, lowercase'],
    'FAISS Embeddings': ['text-embedding-3-small', 'text-embedding-3-small'],
    'Ensemble Weights': ['N/A (solo FAISS)', '[0.5, 0.5] (default)'],
    'Top-K': [5, 10],
    'Canonical Concepts': ['Attivi', 'Attivi'],
    'Reranking': ['No', 'No'],
    'LLM Temperature': [0.7, 0.7],
    'Modifiche Applicate': [
        'Setup iniziale',
        'Rigenerazione BM25 + preprocessing query + top-k 5→10'
    ]
}

df_config = pd.DataFrame(config_history)

# ===========================
# DATI DOMANDE CRITICHE
# ===========================

critical_questions = {
    'Test ID': [1, 1, 1, 2, 2, 2],
    'Question ID': ['CF_GOLD_102', 'CF_GOLD_116', 'CF_GOLD_123', 'CF_GOLD_102', 'CF_GOLD_115', 'CF_GOLD_121'],
    'Domanda': [
        'Qual è la definizione di delegato?',
        'Cosa mi mostra la funzione "Analisi risposte" nel cruscotto?',
        'Per cosa sta la sigla BDN?',
        'Qual è la definizione di delegato?',
        'Cosa vogliono dire Q1, Q2, Q3, Q4 nei cruscotti?',
        'Quali sono considerati principi attivi critici nell\'ambito della DDDAit?'
    ],
    'Metrica Critica': [
        'Faithfulness=0.33',
        'Faithfulness=0.0',
        'Faithfulness=0.33',
        'Faithfulness=0.33',
        'Answer Relevancy=0.0',
        'Answer Relevancy=0.0'
    ],
    'Status': ['Parzialmente risolto (recall 0→1)', 'Risolto (0→0.5)', 'Risolto (0.33→1.0)', 'Persistente', 'Nuovo outlier', 'Nuovo outlier'],
    'Causa Radice': [
        'LLM non aderisce a definizione documento',
        'Chunk mancante in retrieval',
        'Chunk mancante in retrieval',
        'LLM parafrasatura vs definizione esatta',
        'LLM non comprende domanda statistica',
        'LLM non comprende domanda tecnica'
    ]
}

df_critical = pd.DataFrame(critical_questions)

# ===========================
# CALCOLO DELTA METRICHE
# ===========================

metrics_evolution = {
    'Metrica': ['Faithfulness', 'Context Recall', 'Context Precision', 'Answer Relevancy', 'Answer Correctness', 'Health Score'],
    'Test 1': [0.8307, 0.6300, 0.5608, 0.5921, 0.4660, 21.96],
    'Test 2': [0.8674, 0.7033, 0.5972, 0.6704, 0.5185, 21.28],
    'Δ Assoluto': [0.0367, 0.0733, 0.0364, 0.0783, 0.0525, -0.68],
    'Δ %': ['+4.4%', '+11.6%', '+6.5%', '+13.2%', '+11.3%', '-3.1%'],
    'Trend': ['↑', '↑', '↑', '↑', '↑', '↓']
}

df_evolution = pd.DataFrame(metrics_evolution)

# ===========================
# CREAZIONE FILE EXCEL
# ===========================

file_path = 'validation_tracking_classyfarm.xlsx'

with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
    # Sheet 1: Summary
    df_results.to_excel(writer, sheet_name='Test Results Summary', index=False)
    
    # Sheet 2: Metrics Evolution
    df_evolution.to_excel(writer, sheet_name='Metrics Evolution', index=False)
    
    # Sheet 3: Configuration History
    df_config.to_excel(writer, sheet_name='Configuration History', index=False)
    
    # Sheet 4: Critical Questions
    df_critical.to_excel(writer, sheet_name='Critical Questions', index=False)

# ===========================
# FORMATTAZIONE AVANZATA
# ===========================

wb = openpyxl.load_workbook(file_path)

# --- SHEET 1: Test Results Summary ---
ws1 = wb['Test Results Summary']

# Header styling
header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF", size=11)

for cell in ws1[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

# Auto-adjust column widths
column_widths = {
    'A': 25,  # Dataset
    'B': 10,  # N° Test
    'C': 20,  # Data Test
    'D': 50,  # Configurazioni
    'E': 18,  # Faithfulness
    'F': 20,  # Context Recall
    'G': 22,  # Context Precision
    'H': 22,  # Answer Relevancy
    'I': 24,  # Answer Correctness
    'J': 15,  # Health Score
    'K': 80   # Note
}

for col, width in column_widths.items():
    ws1.column_dimensions[col].width = width

# Row height for data rows
for row in ws1.iter_rows(min_row=2, max_row=ws1.max_row):
    ws1.row_dimensions[row[0].row].height = 100
    for cell in row:
        cell.alignment = Alignment(horizontal='left', vertical='top', wrap_text=True)

# Conditional formatting per Health Score
for row in range(2, ws1.max_row + 1):
    cell = ws1[f'J{row}']
    score = cell.value
    if score:
        if score < 20:
            cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # Rosso
        elif score < 40:
            cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")  # Giallo
        else:
            cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # Verde

# --- SHEET 2: Metrics Evolution ---
ws2 = wb['Metrics Evolution']

for cell in ws2[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center')

# Formatting trend cells
for row in range(2, ws2.max_row + 1):
    trend_cell = ws2[f'F{row}']
    delta_cell = ws2[f'E{row}']
    
    if trend_cell.value == '↑':
        trend_cell.font = Font(color="00B050", bold=True, size=14)
        delta_cell.font = Font(color="00B050", bold=True)
    elif trend_cell.value == '↓':
        trend_cell.font = Font(color="C00000", bold=True, size=14)
        delta_cell.font = Font(color="C00000", bold=True)

# --- SHEET 3: Configuration History ---
ws3 = wb['Configuration History']

for cell in ws3[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

for row in ws3.iter_rows(min_row=2):
    for cell in row:
        cell.alignment = Alignment(wrap_text=True, vertical='top')

ws3.column_dimensions['K'].width = 60  # Modifiche Applicate

# --- SHEET 4: Critical Questions ---
ws4 = wb['Critical Questions']

for cell in ws4[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

# Conditional formatting per Status
status_colors = {
    'Risolto': "C6EFCE",          # Verde
    'Parzialmente risolto': "FFEB9C",  # Giallo
    'Persistente': "FFC7CE",      # Rosso
    'Nuovo outlier': "FFC7CE"     # Rosso
}

for row in range(2, ws4.max_row + 1):
    status_cell = ws4[f'E{row}']
    status = status_cell.value
    if status in status_colors:
        status_cell.fill = PatternFill(start_color=status_colors[status], 
                                       end_color=status_colors[status], 
                                       fill_type="solid")

ws4.column_dimensions['C'].width = 60  # Domanda
ws4.column_dimensions['E'].width = 40  # Status
ws4.column_dimensions['F'].width = 50  # Causa

# Salva workbook
wb.save(file_path)

print(f"✅ File Excel creato: {file_path}")
