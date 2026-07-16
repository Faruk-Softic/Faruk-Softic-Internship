import csv
from pathlib import Path

import pandas as pd


CSV_PATH = Path("outputs/run-25/combined_with_regrades.csv")
OUT_PATH = Path("outputs/run-25/combined_with_regrades.token_breakdown.csv")


# Token counting

try:
    import tiktoken
    # Use the tokenizer that matches your runs; nano is correct for gpt-5-nano
    enc = tiktoken.encoding_for_model("gpt-5-nano")

    def count_tokens(text: str) -> int:
        if not text:
            return 0
        return len(enc.encode(text))

except ImportError:
    def count_tokens(text: str) -> int:
        """Crude fallback: ~4 characters per token."""
        text = text or ""
        return max(1, len(text) // 4)

# Load CSV and compute token breakdowns

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    print(f"Reading data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Columns that contain explanatory text from the LLM
    explanation_cols = [
        "reasoning_summary",
        "intro_feedback",
        "methods_feedback",
        "results_feedback",
        "discussion_feedback",
        "lang_style_feedback",
    ]

    # Keep only columns that actually exist
    explanation_cols = [c for c in explanation_cols if c in df.columns]
    print("Using explanation columns:", explanation_cols)

    rows = []

    for idx, row in df.iterrows():
        total_out = row.get("output_tokens", None)

        # Concatenate all explanation/feedback text for this call
        explanation_text = ""
        for c in explanation_cols:
            val = row.get(c, "")
            if isinstance(val, float) and pd.isna(val):
                continue
            explanation_text += str(val)

        explanation_tokens = count_tokens(explanation_text)

        non_expl_tokens = None
        if total_out is not None:
            non_expl_tokens = max(0, total_out - explanation_tokens)

        rows.append(
            {
                "row_index": idx,
                "student_id": row.get("student_id"),
                "pipeline_id": row.get("pipeline_id"),
                "repetition": row.get("repetition"),
                "output_tokens_reported": total_out,
                "approx_explanation_tokens": explanation_tokens,
                "approx_non_explanation_tokens": non_expl_tokens,
            }
        )

# Write summary CSV with token breakdowns

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_index",
                "student_id",
                "pipeline_id",
                "repetition",
                "output_tokens_reported",
                "approx_explanation_tokens",
                "approx_non_explanation_tokens",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote token breakdown to: {OUT_PATH}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()