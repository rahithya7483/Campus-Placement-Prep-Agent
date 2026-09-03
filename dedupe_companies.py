"""
dedupe_companies.py

Tiered company-name de-duplication for placement_cleaned.csv.

Tier 1 (auto-merge, high confidence): exact match after stripping
case/punctuation - e.g. "BYJU's" / "Byjus" / "BYJU`S".

Tier 2 (auto-merge, high confidence): exact match after ALSO stripping
common legal suffixes (Pvt, Pvt Ltd, Limited, Ltd, Company, Corp, Inc) -
e.g. "Sai Life Sciences" / "Sai Life Sciences Limited".

Tier 3 (manual review required): known acronym/full-name pairs, from a
small hand-maintained alias dict (TCS -> Tata Consultancy Services, etc).
Add more pairs here as you find them.

Anything not caught by tiers 1-3 is NOT auto-merged - fuzzy string
similarity alone produces too many false positives (e.g. "Vee Technologies"
vs "Neudesic Technologies" share a word but are different companies).
Those are written to review_candidates.csv for you to eyeball and add to
MANUAL_ALIASES by hand if they're real duplicates.
"""

import os
import re
import pandas as pd
from difflib import SequenceMatcher

IN_PATH = "data/processed/placement_cleaned.csv"
OUT_PATH = "data/processed/placement_cleaned.csv"  # overwrite in place
REVIEW_PATH = "data/processed/company_merge_review_candidates.csv"

LEGAL_SUFFIXES = [
    r"\bpvt\.?\s*ltd\.?\b", r"\bpvt\.?\b", r"\bltd\.?\b", r"\blimited\b",
    r"\bcompany\b", r"\bcorp\.?\b", r"\binc\.?\b",
]

# Tier 3: hand-verified acronym/full-name pairs. Each key maps to the
# canonical (full) name. Add more as you discover them.
MANUAL_ALIASES = {
    "tcs": "Tata Consultancy Services",
    "sls": "Sai Life Sciences",
    "srisahintechsolutionsssts": "Sri Sahin Tech Solutions",
    "bajajfin": "Bajaj Finance",
    "medhoservodrives": "Medha Servo Drives",
    "medhoservo": "Medha Servo Drives",
    "medhaservo": "Medha Servo Drives",
    "esaf": "ESAF Small Finance Bank",
    "esafsfb": "ESAF Small Finance Bank",
    "funnelmovingdneedle": "Funnl",
    "movindneedle": "Funnl",
}


def norm_basic(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def norm_no_suffix(s: str) -> str:
    s = s.lower()
    for pat in LEGAL_SUFFIXES:
        s = re.sub(pat, "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def build_canonical_map(companies: list[str]) -> dict[str, str]:
    canonical = {}

    # Tier 1: exact-after-basic-normalize groups -> canonical = most frequent spelling
    from collections import Counter
    counts = Counter(companies)
    groups_basic = {}
    for c in set(companies):
        groups_basic.setdefault(norm_basic(c), []).append(c)
    for key, variants in groups_basic.items():
        if len(variants) > 1:
            best = max(variants, key=lambda v: (counts[v], len(v)))
            for v in variants:
                canonical[v] = best

    # Tier 2: exact-after-suffix-stripped groups (only for names not already merged)
    remaining = [c for c in set(companies) if c not in canonical]
    groups_suffix = {}
    for c in remaining:
        groups_suffix.setdefault(norm_no_suffix(c), []).append(c)
    for key, variants in groups_suffix.items():
        if len(variants) > 1:
            best = max(variants, key=lambda v: (counts[v], len(v)))
            for v in variants:
                canonical[v] = best

    # Tier 3: manual aliases
    for c in set(companies):
        key = norm_basic(c)
        if key in MANUAL_ALIASES:
            canonical[c] = MANUAL_ALIASES[key]

    return canonical


def find_review_candidates(companies: list[str], canonical_map: dict) -> list[tuple]:
    already_handled = set(canonical_map.keys())
    unresolved = [c for c in set(companies) if c not in already_handled]
    flagged = []
    for i in range(len(unresolved)):
        for j in range(i + 1, len(unresolved)):
            a, b = unresolved[i], unresolved[j]
            ratio = SequenceMatcher(None, norm_basic(a), norm_basic(b)).ratio()
            if ratio > 0.72 and abs(len(norm_basic(a)) - len(norm_basic(b))) > 2:
                flagged.append((round(ratio, 2), a, b))
    flagged.sort(reverse=True)
    return flagged


def main():
    df = pd.read_csv(IN_PATH)
    companies = df["clean_company"].tolist()

    canonical_map = build_canonical_map(companies)
    n_merged_rows = sum(1 for c in companies if c in canonical_map)
    n_groups = len(set(canonical_map.values()) & set(canonical_map.keys()) | set(canonical_map.values()))

    print(f"Tier 1+2+3 merges: {len(canonical_map)} distinct spellings collapsed")
    print(f"Rows affected: {n_merged_rows} / {len(df)}")

    df["clean_company"] = df["clean_company"].apply(lambda c: canonical_map.get(c, c))

    tmp_path = OUT_PATH + ".tmp"
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, OUT_PATH)  # atomic swap - only overwrites OUT_PATH if the write above succeeded
    except PermissionError as e:
        print(f"\nCouldn't write to {OUT_PATH}: {e}")
        print("This is almost always because the file is open in Excel/another program, "
              "or is marked read-only. Close it / unlock it and re-run this script.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    print(f"Saved de-duplicated data to {OUT_PATH}")

    review = find_review_candidates(companies, canonical_map)
    pd.DataFrame(review, columns=["similarity", "company_a", "company_b"]).to_csv(REVIEW_PATH, index=False)
    print(f"\n{len(review)} additional candidate pairs need manual review -> {REVIEW_PATH}")
    print("(These were NOT auto-merged - check them, and add real duplicates to MANUAL_ALIASES)")


if __name__ == "__main__":
    main()