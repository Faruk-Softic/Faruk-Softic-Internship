"""
Reads a run output JSONL and writes a summary CSV with token counts,
reply lengths, and reasoning summary lengths per request.

Usage:
    python extract_run_summary.py experiments/run-25/run-25_output.jsonl
"""

import json
import csv
import sys
from pathlib import Path


def extract_reply_and_reasoning(response_body: dict) -> tuple[str, str]:
    message_text = ""
    reasoning_summary = ""
    for item in response_body.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    message_text = block["text"]
        elif item.get("type") == "reasoning":
            blocks = [
                b["text"]
                for b in item.get("summary", [])
                if b.get("type") == "summary_text"
            ]
            reasoning_summary = "\n\n".join(blocks)
    return message_text, reasoning_summary


def main(jsonl_path: Path):
    rows = []

    for line in jsonl_path.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)
        custom_id = entry.get("custom_id", "unknown")
        error = entry.get("error")

        parts = custom_id.split("__")
        run_label   = parts[0] if len(parts) > 0 else ""
        student_id  = parts[1] if len(parts) > 1 else ""
        pipeline_id = parts[2] if len(parts) > 2 else ""
        repetition  = parts[3] if len(parts) > 3 else "run1"

        if error:
            rows.append({
                "custom_id":               custom_id,
                "run_label":               run_label,
                "student_id":              student_id,
                "pipeline_id":             pipeline_id,
                "repetition":              repetition,
                "status":                  "error",
                "input_tokens":            None,
                "output_tokens":           None,
                "reasoning_summary_chars": None,
                "reply_chars":             None,
                "reply_valid_json":        None,
                "error":                   str(error),
            })
            continue

        response_body = entry.get("response", {}).get("body", {})
        usage = response_body.get("usage", {})
        input_tokens  = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        reply, reasoning = extract_reply_and_reasoning(response_body)

        try:
            json.loads(reply)
            valid_json = True
        except Exception:
            valid_json = False

        rows.append({
            "custom_id":               custom_id,
            "run_label":               run_label,
            "student_id":              student_id,
            "pipeline_id":             pipeline_id,
            "repetition":              repetition,
            "status":                  "ok",
            "input_tokens":            input_tokens,
            "output_tokens":           output_tokens,
            "reasoning_summary_chars": len(reasoning),
            "reply_chars":             len(reply),
            "reply_valid_json":        valid_json,
            "error":                   "",
        })

    out_path = jsonl_path.parent / (jsonl_path.stem + "_summary.csv")
    fieldnames = [
        "custom_id", "run_label", "student_id", "pipeline_id", "repetition",
        "status", "input_tokens", "output_tokens",
        "reasoning_summary_chars", "reply_chars", "reply_valid_json", "error",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary statistics
    ok_rows = [r for r in rows if r["status"] == "ok"]
    err_rows = [r for r in rows if r["status"] == "error"]

    print(f"\nTotal requests : {len(rows)}")
    print(f"  OK           : {len(ok_rows)}")
    print(f"  Errors       : {len(err_rows)}")

    if ok_rows:
        total_in  = sum(r["input_tokens"]  or 0 for r in ok_rows)
        total_out = sum(r["output_tokens"] or 0 for r in ok_rows)
        avg_in    = total_in  / len(ok_rows)
        avg_out   = total_out / len(ok_rows)
        avg_rsm   = sum(r["reasoning_summary_chars"] or 0 for r in ok_rows) / len(ok_rows)
        avg_reply = sum(r["reply_chars"] or 0 for r in ok_rows) / len(ok_rows)
        invalid   = sum(1 for r in ok_rows if not r["reply_valid_json"])

        print(f"\nToken totals:")
        print(f"  Total input tokens  : {total_in:,}")
        print(f"  Total output tokens : {total_out:,}")
        print(f"\nPer-request averages:")
        print(f"  Avg input tokens    : {avg_in:,.0f}")
        print(f"  Avg output tokens   : {avg_out:,.0f}")
        print(f"  Avg reasoning chars : {avg_rsm:,.0f}")
        print(f"  Avg reply chars     : {avg_reply:,.0f}")
        print(f"\nInvalid JSON replies : {invalid}")

        # Breakdown by pipeline
        print(f"\nBreakdown by pipeline:")
        for pid in sorted(set(r["pipeline_id"] for r in ok_rows)):
            p_rows = [r for r in ok_rows if r["pipeline_id"] == pid]
            p_out  = sum(r["output_tokens"] or 0 for r in p_rows)
            p_rsm  = sum(r["reasoning_summary_chars"] or 0 for r in p_rows) / len(p_rows)
            p_rep  = sum(r["reply_chars"] or 0 for r in p_rows) / len(p_rows)
            print(f"  {pid}: {len(p_rows)} requests | "
                  f"total output tokens: {p_out:,} | "
                  f"avg reasoning chars: {p_rsm:,.0f} | "
                  f"avg reply chars: {p_rep:,.0f}")

    print(f"\nSummary CSV written to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_run_summary.py <path_to_output.jsonl>")
        sys.exit(1)
    main(Path(sys.argv[1]))
