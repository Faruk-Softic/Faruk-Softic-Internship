"""
This script merges regrading data from five Excel files from five regraders with the combined.csv file.
"""

import pandas as pd
from pathlib import Path

# ── Edit these paths ──────────────────────────────────────────────────────────
COMBINED_CSV   = Path("outputs/combined.csv")
REGRADE_FOLDER = Path("data/Regrading")
OUTPUT_CSV     = Path("outputs/combined_with_regrades.csv")
# ─────────────────────────────────────────────────────────────────────────────

SECTION_WEIGHTS = {
    "intro":      0.30,
    "methods":    0.15,
    "results":    0.15,
    "discussion": 0.30,
    "lang_style": 0.10,
}

# Expected column names in the Excel files → internal section keys
REGRADE_COLS = {
    "student_id":       "student_id",   # passthrough
    "introduction":     "intro",
    "methods":          "methods",
    "results":          "results",
    "discussion":       "discussion",
    "language_style":   "lang_style",
}


def compute_final_grade(row: pd.Series) -> float:
    """Compute weighted final grade from internal section key columns."""
    return sum(row[sec] * w for sec, w in SECTION_WEIGHTS.items())


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/strip column names and fix known quirks (e.g. 'language_ style')."""
    fixed = []
    for c in df.columns:
        c2 = c.strip().lower()
        if c2 == "language_ style":  # grader-2 typo
            c2 = "language_style"
        fixed.append(c2)
    df.columns = fixed
    return df


def load_regrades(folder: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(folder.glob("*.xlsx")):
        df = pd.read_excel(path)
        df = normalise_columns(df)

        # Check for missing expected columns BEFORE doing anything else
        missing = [c for c in REGRADE_COLS if c not in df.columns]
        if missing:
            print(f"  ⚠  '{path.name}': missing columns {missing} — skipping file")
            print(f"       Columns found: {list(df.columns)}")
            continue

        grade_src_cols = ["introduction", "methods", "results", "discussion", "language_style"]
        keep_cols = ["student_id"] + grade_src_cols
        df = df[keep_cols].copy()

        df = df.dropna(subset=grade_src_cols)
        if len(df) == 0:
            print(f"  '{path.name}': 0 graded rows after dropping empty rows — skipping")
            continue

        df = df.rename(columns=REGRADE_COLS)

        for sec in SECTION_WEIGHTS.keys():
            df[sec] = pd.to_numeric(df[sec], errors="coerce")
        df = df.dropna(subset=list(SECTION_WEIGHTS.keys()))
        if len(df) == 0:
            print(f"  '{path.name}': 0 graded rows after numeric conversion — skipping")
            continue

        # Compute weighted final grade
        df["regrade_final"] = df.apply(compute_final_grade, axis=1).round(2)

        frames.append(df)
        print(f"  Loaded '{path.name}': {len(df)} graded rows")

    if not frames:
        raise ValueError("No valid regrading data found. Check column names and grade values in Excel files.")

    combined = pd.concat(frames, ignore_index=True)

    # Handle duplicate student_ids (graded by more than one regrader)
    dupes = combined[combined.duplicated("student_id", keep=False)]
    if not dupes.empty:
        dup_ids = dupes["student_id"].unique().tolist()
        print(f"\n  ⚠  Students graded by multiple regraders: {dup_ids}")
        print(f"     Averaging their grades.")
        grade_cols = list(SECTION_WEIGHTS.keys()) + ["regrade_final"]
        combined = combined.groupby("student_id")[grade_cols].mean().round(2).reset_index()

    # Rename section columns to regrade_ prefix
    for sec in SECTION_WEIGHTS:
        combined = combined.rename(columns={sec: f"regrade_{sec}"})

    print(f"\n  Total regraded students: {len(combined)}")
    return combined


def main():
    print("Loading combined.csv...")
    combined = pd.read_csv(COMBINED_CSV)
    print(f"  {len(combined)} rows, {combined['student_id'].nunique()} unique students")

    print("\nLoading regrading files...")
    regrades = load_regrades(REGRADE_FOLDER)

    print("\nMerging...")
    merged = combined.merge(regrades, on="student_id", how="left")

    n_matched = merged["regrade_final"].notna().sum()
    n_missing = merged["regrade_final"].isna().sum()
    print(f"  Rows with regrade:    {n_matched}")
    print(f"  Rows without regrade: {n_missing}")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✓ Saved to '{OUTPUT_CSV}'")

    # Summary of regrade grades (one row per student)
    regrade_students = merged[merged["regrade_final"].notna()].drop_duplicates("student_id")
    print(f"\nRegrade final grade summary ({len(regrade_students)} students):")
    print(regrade_students["regrade_final"].describe().round(2))


if __name__ == "__main__":
    main()
