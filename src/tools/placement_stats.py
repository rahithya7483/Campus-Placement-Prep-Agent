
import sqlite3
import os
import re
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "placement.db")

FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|ATTACH|DETACH|PRAGMA|CREATE|REPLACE|TRIGGER|VACUUM)\b",
    re.IGNORECASE,
)

SCHEMA_DESC = """
Table: placement_stats
Columns:
- year (TEXT) academic year in the exact format 'YYYY-YY', e.g. '2016-17', '2024-25'.
  IMPORTANT: always normalize the user's year phrasing to this exact format before using
  it in a WHERE clause. '2024-2025', '2024/25', 'academic year 2024-25', and '2024-25' all
  refer to the SAME row and must be written as the literal string '2024-25' in SQL - never
  '2024-2025'.
- raw_company_name (TEXT) original messy name as scraped - don't use for filtering
- clean_company (TEXT) normalized, DE-DUPLICATED company name - there is exactly ONE
  canonical spelling per company (e.g. always 'Tata Consultancy Services', never 'TCS').
  When the user uses a well-known abbreviation or short form, resolve it to the canonical
  full name before filtering. ALWAYS filter/group on this column, never raw_company_name.
- n_selected (INTEGER, may be NULL) number of students selected in that row
- ctc_lpa (REAL, may be NULL) package offered, in LPA (Lakhs Per Annum)
- ctc_needs_review (INTEGER 0/1) whether the CTC value is uncertain

Note: each row is one (company, year, package-tier) combination - a company
can have multiple rows in the same year if it offered different packages to
different students. To get a company's TOTAL selected in a year, SUM(n_selected).
"""

SQL_PROMPT_TEMPLATE = f"""You write SQLite SELECT queries to answer questions about
college placement data. Only ever write a single read-only SELECT statement.

{SCHEMA_DESC}

IMPORTANT: always match clean_company case-insensitively using "COLLATE NOCASE" in the
WHERE clause, since you cannot be certain of a company's exact stored capitalization
(e.g. write clean_company = 'byju''s' COLLATE NOCASE, not a guessed exact-case match).

Examples:
Q: What was the highest package offered in 2016-17?
SQL: SELECT clean_company, MAX(ctc_lpa) as max_ctc FROM placement_stats WHERE year = '2016-17' AND ctc_lpa IS NOT NULL;

Q: What was the highest package in 2024-2025?
SQL: SELECT clean_company, MAX(ctc_lpa) as max_ctc FROM placement_stats WHERE year = '2024-25' AND ctc_lpa IS NOT NULL;

Q: How many companies visited in the year 2024-2025?
SQL: SELECT COUNT(DISTINCT clean_company) as n_companies FROM placement_stats WHERE year = '2024-25';

Q: What package did byjus offer?
SQL: SELECT clean_company, year, ctc_lpa FROM placement_stats WHERE clean_company = 'byju''s' COLLATE NOCASE AND ctc_lpa IS NOT NULL ORDER BY year;

Q: Compare TCS and Infosys packages
SQL: SELECT clean_company, year, ctc_lpa, n_selected FROM placement_stats WHERE clean_company = 'Tata Consultancy Services' COLLATE NOCASE OR clean_company = 'Infosys' COLLATE NOCASE ORDER BY clean_company, year;

Q: Has EPAM visited more than once in the same year?
SQL: SELECT year, COUNT(*) as n_rows FROM placement_stats WHERE clean_company = 'EPAM' COLLATE NOCASE GROUP BY year HAVING COUNT(*) > 1;

Now write the SQL for this question. Respond with ONLY the SQL query, no explanation, no markdown fences.

Q: {{question}}
SQL:
"""

SYNTH_PROMPT_TEMPLATE = """Answer the user's question using ONLY the SQL query result below.
Be concise and precise with numbers. If the result is empty, say the data doesn't
have an answer - don't guess.

Question: {question}
SQL query used: {sql}
Result rows: {rows}
{assumption_note}
Answer:
"""


def is_safe_select(sql: str) -> bool:
    stripped = sql.strip().rstrip(";")
    if not stripped.upper().startswith("SELECT"):
        return False
    if ";" in stripped:  # no stacked statements
        return False
    if FORBIDDEN.search(stripped):
        return False
    return True


def run_query(sql: str) -> list[dict]:
    # read-only connection - a second, independent safety layer beyond the
    # keyword check, in case a forbidden pattern slips past the regex
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def query_placement_stats(question: str, llm_call) -> dict:
    prompt = SQL_PROMPT_TEMPLATE.format(question=question)
    sql = llm_call(prompt).strip()
    sql = re.sub(r"^```sql|```$|^```", "", sql, flags=re.MULTILINE).strip()

    if not is_safe_select(sql):
        return {"error": f"Generated query rejected for safety: {sql}", "sql": sql, "rows": []}

    try:
        rows = run_query(sql)
    except Exception as e:
        return {"error": f"Query execution failed: {e}", "sql": sql, "rows": []}

    return {"sql": sql, "rows": rows, "error": None}


def is_effectively_empty(rows: list[dict]) -> bool:
    if not rows:
        return True
    return all(v is None for row in rows for v in row.values())


RANGE_PATTERN = re.compile(r"\b(19|20)\d{2}\s*[-/]\s*\d{2,4}\b")
BARE_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
YEAR_LITERAL_RE = re.compile(r"year\s*=\s*'((?:19|20)\d{2}-\d{2})'", re.IGNORECASE)


def question_has_ambiguous_year(question: str) -> bool:
    """True if the question mentions a bare 4-digit year (e.g. '2025') without
    ever giving a proper academic-year range anywhere in the question."""
    if RANGE_PATTERN.search(question):
        return False
    return bool(BARE_YEAR_PATTERN.search(question))


def extract_year_used(sql: str) -> str | None:
    """Pulls the actual 'YYYY-YY' the generated SQL filtered on, if any."""
    m = YEAR_LITERAL_RE.search(sql)
    return m.group(1) if m else None

def answer_stats_question(question: str, llm_call) -> dict:
    result = query_placement_stats(question, llm_call)
    if result["error"]:
        return {"answer": f"Couldn't answer that from the data ({result['error']}).", "sql": result["sql"], "rows": []}

    if is_effectively_empty(result["rows"]):
        return {"answer": "No data is recorded for that.", "sql": result["sql"], "rows": result["rows"]}

    assumption_note = ""
    if question_has_ambiguous_year(question):
        year_used = extract_year_used(result["sql"])
        if year_used:
            assumption_note = (
                f"Note: the question gave a single year without an academic-year range, so this "
                f"was interpreted as the {year_used} academic year. Explicitly mention this "
                f'assumption in your answer (e.g. "assuming the {year_used} academic year...").\n'
            )

    synth_prompt = SYNTH_PROMPT_TEMPLATE.format(
        question=question, sql=result["sql"], rows=json.dumps(result["rows"]), assumption_note=assumption_note
    )
    answer = llm_call(synth_prompt)
    return {"answer": answer, "sql": result["sql"], "rows": result["rows"]}


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from llm_client import get_llm_call

    for q in [
        "What was the highest package offered in 2016-17?",
        "Compare TCS and Infosys packages",
        "What package did byjus offer?",
    ]:
        print(f"\nQ: {q}")
        print(json.dumps(answer_stats_question(q, get_llm_call()), indent=2))