from __future__ import annotations

import csv
import os
from typing import List, Dict, Optional


def _csv_path() -> str:
    here = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(here, "data", "jp_tickers.csv")


def list_jp_tickers(query: Optional[str] = None, limit: int = 200) -> List[Dict]:
    path = _csv_path()
    rows: List[Dict] = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({"ticker": r["ticker"], "name": r["name"], "sector": r["sector"]})
    if query:
        q = query.strip().lower()
        rows = [r for r in rows if q in r["ticker"].lower() or q in r["name"].lower() or q in r["sector"].lower()]
    return rows[:limit]

