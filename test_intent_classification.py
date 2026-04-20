"""Test classificazione intent migliorata"""

import sys
from pathlib import Path

BASE_DIR = Path('.').resolve()
sys.path.insert(0, str(BASE_DIR))

# Import dopo setup path
from core.layers.layer8_application import infer_intent

print('='*60)
print('TEST INTENT CLASSIFICATION')
print('='*60)

# Test cases con expected intent
test_cases = [
    # Procedural
    ("Come faccio a scaricare il report SQNBA?", "procedural"),
    ("Come richiedo il profilo da operatore?", "procedural"),
    ("Come faccio ad associare un valutatore?", "procedural"),
    ("Dove trovo gli allevamenti?", "procedural"),
    
    # Acronym definition
    ("Per cosa sta BDN?", "acronym_definition"),
    ("Cosa si intende per OdC?", "acronym_definition"),
    ("Cosa significa SQNBA?", "acronym_definition"),
    
    # Definitional
    ("Cosa vogliono dire Q1, Q2, Q3, Q4 nei cruscotti?", "definitional"),
    ("Qual è la definizione di operatore?", "definitional"),
    ("Cos'è un delegato?", "definitional"),
    
    # Generic
    ("Informazioni sul sistema", "generic"),
]

print('\nTest domande:\n')

correct = 0
total = len(test_cases)

for question, expected in test_cases:
    result = infer_intent(question)
    is_correct = result == expected
    
    if is_correct:
        correct += 1
        status = "✅"
    else:
        status = "❌"
    
    print(f'{status} "{question}"')
    print(f'   Expected: {expected}')
    print(f'   Got: {result}')
    
    if not is_correct:
        print(f'   ⚠️  MISMATCH!')
    
    print()

print('='*60)
print(f'RISULTATO: {correct}/{total} corretti ({correct/total*100:.1f}%)')
print('='*60)

if correct == total:
    print('✅ FIX 2 INTENT CLASSIFICATION COMPLETATO!\n')
    print('Prossimo step: Test comparativo finale')
else:
    print(f'⚠️  {total - correct} classificazioni errate')

print('='*60)