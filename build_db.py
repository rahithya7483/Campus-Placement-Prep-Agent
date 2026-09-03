"""
build_db.py

Builds data/processed/placement.db from the two cleaned CSVs.
Run once (and re-run any time the CSVs change).

Two tables:
- placement_stats: one row per (company, year, ctc tier) - the numeric data
- company_notes: one row per company with detailed interview info

The SQL tool only ever queries placement_stats (numbers/aggregates).
The RAG tool keeps using ChromaDB for company_notes (qualitative text).
"""

import sqlite3
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "processed", "placement.db")


def main():
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)

    placement = pd.read_csv(os.path.join(DATA_DIR, "processed", "placement_cleaned.csv"))
    notes = pd.read_csv(os.path.join(DATA_DIR, "raw", "company_notes_placement_agent.csv"))
    notes.columns = [c.strip() for c in notes.columns]

    conn = sqlite3.connect(DB_PATH)
    placement.to_sql("placement_stats", conn, if_exists="replace", index=False)
    notes.to_sql("company_notes", conn, if_exists="replace", index=False)

    # sanity check
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM placement_stats")
    print(f"placement_stats rows: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(*) FROM company_notes")
    print(f"company_notes rows: {cur.fetchone()[0]}")
    cur.execute("PRAGMA table_info(placement_stats)")
    print("placement_stats columns:", [row[1] for row in cur.fetchall()])

    conn.close()
    print(f"Saved to {DB_PATH}")


if __name__ == "__main__":
    main()