# File: test_acronym_detection.py

import re

test_questions = [
    "Per cosa sta BDN?",
    "Cosa si intende per OdC?",
    "Cosa significa SQNBA?",
]

pattern = r'\bper\s+cosa\s+sta\s+([A-Z]{2,6})\b'

for q in test_questions:
    print(f'\nQuestion: "{q}"')
    
    # Test pattern
    match = re.search(pattern, q, re.IGNORECASE)
    print(f'  Pattern match: {match}')
    
    if match:
        print(f'  Groups: {match.groups()}')
        print(f'  Matched text: {match.group(0)}')
        if match.groups():
            acronym = match.group(1)
            print(f'  Acronym: {acronym}')
            print(f'  Is upper: {acronym.isupper()}')
    
    # Test tutti i pattern
    patterns = [
        r'\bper\s+cosa\s+sta\s+([A-Z]{2,6})\b',
        r'\bcosa\s+significa\s+([A-Z]{2,6})\b',
        r'\bcosa\s+si\s+intende\s+per\s+([A-Z]{2,6})\b',
    ]
    
    for i, p in enumerate(patterns):
        m = re.search(p, q, re.IGNORECASE)
        if m:
            print(f'  ✅ Pattern {i+1} matched!')