import json
import re
import requests
import os
from typing import Any, Dict, List, Optional

def load_config(path: str = "validation/config.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        # Fallback path try
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "config.json")
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Config file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"⚠️  Dataset non trovato: {path}")
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def safe_post(url: str, payload: Dict[str, Any], timeout: int = 30) -> Optional[Dict[str, Any]]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        if not r.text:
            return None
        return r.json()
    except Exception as e:
        print(f"⚠️  API Error ({url}): {e}")
        return None

def extract_first_number(text: str) -> Optional[float]:
    if text is None: return None
    matches = re.findall(r"[-+]?\d*\.?\d+", str(text))
    return float(matches[0]) if matches else None