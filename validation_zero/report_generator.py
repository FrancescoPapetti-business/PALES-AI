"""
Generatore report HTML per validazione zero-knowledge
"""
from typing import Dict, Any
from pathlib import Path
from datetime import datetime


def generate_html_report(report: Dict[str, Any], output_path: Path):
    """
    Genera report HTML interattivo con grafici e dettagli
    
    Args:
        report: Dict con risultati validazione
        output_path: Path dove salvare HTML
    """
    
    # Estrai dati principali
    test_name = report['test_name']
    timestamp = report['timestamp']
    health_score = report['health_score']
    metrics_agg = report['metrics_aggregate']
    status_dist = report['status_distribution']
    critical_questions = report['critical_questions']
    
    # Template HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report Validazione - {test_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        
        .health-score {{
            background: white;
            color: #667eea;
            display: inline-block;
            padding: 20px 40px;
            border-radius: 50px;
            font-size: 3em;
            font-weight: bold;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        }}
        
        .metric-card h3 {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-value.good {{
            color: #28a745;
        }}
        
        .metric-value.warning {{
            color: #ffc107;
        }}
        
        .metric-value.danger {{
            color: #dc3545;
        }}
        
        .metric-details {{
            font-size: 0.9em;
            color: #666;
            margin-top: 10px;
        }}
        
        .status-bar {{
            display: flex;
            height: 40px;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
        }}
        
        .status-segment {{
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: flex-grow 0.3s;
        }}
        
        .status-segment:hover {{
            flex-grow: 1.1 !important;
        }}
        
        .status-excellent {{
            background: #28a745;
        }}
        
        .status-good {{
            background: #17a2b8;
        }}
        
        .status-moderate {{
            background: #ffc107;
        }}
        
        .status-critical {{
            background: #dc3545;
        }}
        
        .critical-questions {{
            margin-top: 20px;
        }}
        
        .question-card {{
            background: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }}
        
        .question-card.critical {{
            background: #f8d7da;
            border-left-color: #dc3545;
        }}
        
        .question-id {{
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .question-text {{
            font-size: 1.1em;
            margin-bottom: 10px;
        }}
        
        .question-meta {{
            font-size: 0.9em;
            color: #666;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #dee2e6;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        th {{
            background: #667eea;
            color: white;
            font-weight: bold;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Report Validazione ClassyFarm RAG</h1>
            <p class="subtitle">{test_name}</p>
            <p class="subtitle">Timestamp: {timestamp}</p>
            <div class="health-score">{health_score:.1f}%</div>
            <p style="margin-top: 10px; font-size: 1.2em;">Health Score</p>
        </div>
        
        <div class="content">
            <!-- Metriche Principali -->
            <div class="section">
                <h2>📈 Metriche Aggregate</h2>
                <div class="metrics-grid">
"""
    
    # Aggiungi card per ogni metrica
    for metric_name, metric_data in metrics_agg.items():
        mean_val = metric_data['mean']
        
        # Determina colore basato su threshold
        if metric_name in report['thresholds']:
            threshold = report['thresholds'][metric_name]
            if mean_val >= threshold:
                color_class = "good"
            elif mean_val >= threshold - 0.1:
                color_class = "warning"
            else:
                color_class = "danger"
        else:
            color_class = ""
        
        html_content += f"""
                    <div class="metric-card">
                        <h3>{metric_name.replace('_', ' ').title()}</h3>
                        <div class="metric-value {color_class}">{mean_val:.3f}</div>
                        <div class="metric-details">
                            Range: {metric_data['min']:.3f} - {metric_data['max']:.3f}<br>
"""
        
        if 'below_threshold' in metric_data:
            html_content += f"                            Sotto soglia: {metric_data['below_threshold']}<br>\n"
        
        html_content += "                        </div>\n"
        html_content += "                    </div>\n"
    
    html_content += """
                </div>
            </div>
            
            <!-- Distribuzione Status -->
            <div class="section">
                <h2>📊 Distribuzione Status</h2>
                <div class="status-bar">
"""
    
    # Calcola percentuali per status bar
    total = sum(status_dist.values())
    for status, count in status_dist.items():
        percentage = (count / total * 100) if total > 0 else 0
        status_class = f"status-{status.lower()}"
        html_content += f"""
                    <div class="status-segment {status_class}" style="flex-grow: {percentage};">
                        {status}: {count} ({percentage:.1f}%)
                    </div>
"""
    
    html_content += """
                </div>
            </div>
            
            <!-- Domande Critiche -->
            <div class="section">
                <h2>🚨 Domande Critiche/Moderate</h2>
"""
    
    if critical_questions:
        html_content += f"                <p>Trovate <strong>{len(critical_questions)}</strong> domande con problemi:</p>\n"
        html_content += "                <div class=\"critical-questions\">\n"
        
        for q in critical_questions:
            card_class = "critical" if q['status'] == 'CRITICAL' else ""
            html_content += f"""
                    <div class="question-card {card_class}">
                        <div class="question-id">{q['question_id']}</div>
                        <div class="question-text">{q['question']}</div>
                        <div class="question-meta">
                            Status: <strong>{q['status']}</strong> | 
                            Critical Flags: <strong>{q['critical_flags']}</strong> | 
                            Metriche fallite: <strong>{', '.join(q['failed_metrics'])}</strong>
                        </div>
                    </div>
"""
        
        html_content += "                </div>\n"
    else:
        html_content += "                <p>✅ Nessuna domanda critica trovata!</p>\n"
    
    html_content += """
            </div>
            
            <!-- Tabella Riepilogo -->
            <div class="section">
                <h2>📋 Riepilogo per Domanda</h2>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Domanda</th>
                            <th>Groundedness</th>
                            <th>Self-Cont.</th>
                            <th>Relevance</th>
                            <th>Consistency</th>
                            <th>Context P.</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Aggiungi righe tabella
    for result in report['detailed_results']:
        html_content += f"""
                        <tr>
                            <td>{result['question_id']}</td>
                            <td>{result['question'][:60]}...</td>
                            <td>{result['metrics']['groundedness']['score']:.3f}</td>
                            <td>{result['metrics']['self_contained']['score']:.3f}</td>
                            <td>{result['metrics']['relevance']['score']:.3f}</td>
                            <td>{result['metrics']['consistency']['score']:.3f}</td>
                            <td>{result['metrics']['context_precision']['score']:.3f}</td>
                            <td><strong>{result['status']}</strong></td>
                        </tr>
"""
    
    html_content += """
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p>Report generato automaticamente da ClassyFarm RAG Validation Framework</p>
            <p>Zero-Knowledge Validation (senza ground truth)</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Salva HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Report HTML salvato: {output_path}")