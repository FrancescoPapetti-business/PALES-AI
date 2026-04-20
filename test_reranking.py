"""
Quick test script for re-ranking functionality.
Tests a single query to verify integration.
"""

import sys
import os
import logging

# Setup path
sys.path.insert(0, os.path.dirname(__file__))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

from core.layers.layer8_application import process_request

# Test query
TEST_QUERY = "Qual è la definizione di operatore?"

print("="*80)
print("TESTING RE-RANKING INTEGRATION")
print("="*80)
print(f"\nQuery: {TEST_QUERY}\n")

try:
    # Process query
    result = process_request(TEST_QUERY)
    
    # Display results
    print("="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Latency: {result['latency']}s")
    print(f"Answer Mode: {result.get('answer_mode', 'unknown')}")
    print(f"BM25 Active: {result.get('bm25_active', False)}")
    
    print("\n" + "="*80)
    print("✅ Test completed successfully!")
    print("="*80)
    
except Exception as e:
    print("\n" + "="*80)
    print("❌ TEST FAILED")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)