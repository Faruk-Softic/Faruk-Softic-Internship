""""
This script runs the batch_input.jsonl file using the OpenAI Batch API, polls for completion, 
and saves the results to a CSV file, skipping any requests that have already been completed 
(based on the run_id in the CSV file). It also counts the total input and output tokens used in the batch run.
"""

import os
import csv
import json
import time
import io
from datetime import datetime
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from config import SECTION_KEYS, CHECKLIST_KEYS

load_dotenv()

BATCH_JSONL   = os.environ.get("BATCH_JSONL", "batch_input.jsonl")
RESULTS_CSV   = os.environ.get("RESULTS_CSV")
POLL_INTERVAL = 60

WEIGHTS = {
    "introduction":   0.30,
    "methods":        0.15,
    "results":        0.15,
    "discussion":     0.30,
    "language_style": 0.10,
}

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Helper functions

def _parse_grade(value) -> Optional[float]:
    try:
        g = float(value)
        return round(g * 2) / 2.0 if 1.0 <= g <= 10.0 else None
    except (TypeError, ValueError):
        return None


def parse_reply(raw: str) -> dict:
    base = {f"{k}_grade": None for k in SECTION_KEYS}
    base.update({f"{k}_feedback": None for k in SECTION_KEYS})
    base.update({k: None for k in CHECKLIST_KEYS})
    base.update({"raw_reply": raw, "missing_sections": ""})

    try:
        data = json.loads(raw)
    except Exception:
        return base

    for k in SECTION_KEYS:
        base[f"{k}_grade"]    = _parse_grade(data.get(f"{k}_grade"))
        base[f"{k}_feedback"] = data.get(f"{k}_feedback")

    for k in CHECKLIST_KEYS:
        val = data.get(k)
        base[k] = bool(val) if val is not None else None

    section_map = {"introduction": "intro", "methods": "methods", "results": "results",
                   "discussion": "discussion", "language_style": "lang_style"}
    missing = [sec for sec, key in section_map.items() if base[f"{key}_grade"] is None]
    base["missing_sections"] = ", ".join(missing)
    return base


def append_result_row(csv_path: str, row: dict) -> None:
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_metadata_from_jsonl(path: str) -> dict:
    """Reconstruct metadata from the JSONL file (custom_id -> pipeline info)."""
    metadata = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            req = json.loads(line)
            cid = req["custom_id"]
            parts = cid.split(".")
            run_part     = parts[-1]
            pipeline_id  = parts[-2]
            student_id   = ".".join(parts[:-2])
            rubric_version = "improved" if pipeline_id.startswith("B") else "original"
            metadata[cid] = {
                "run_id":       cid,
                "student_id":   student_id,
                "pipeline_id":  rubric_version,
                "model":        req["body"]["model"],
                "temperature":  req["body"].get("temperature"),
            }
    return metadata


def process_result(result_line: dict, metadata: dict) -> dict | None:
    custom_id = result_line.get("custom_id", "unknown")
    meta      = metadata.get(custom_id)
    if meta is None:
        print(f"  ⚠  No metadata for '{custom_id}' — skipping.")
        return None
    if error := result_line.get("error"):
        print(f"  ✗ API error for '{custom_id}': {error}")
        return None
    try:
        raw_reply = result_line["response"]["body"]["output"][0]["content"][0]["text"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"  ✗ Could not extract reply for '{custom_id}': {e}")
        return None

    usage         = result_line.get("response", {}).get("body", {}).get("usage", {})
    input_tokens  = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    parsed = parse_reply(raw_reply)
    grades = {
        "introduction":   parsed["intro_grade"],
        "methods":        parsed["methods_grade"],
        "results":        parsed["results_grade"],
        "discussion":     parsed["discussion_grade"],
        "language_style": parsed["lang_style_grade"],
    }
    final_grade = (sum(grades[k] * w for k, w in WEIGHTS.items())
                   if all(v is not None for v in grades.values()) else None)

    if parsed["missing_sections"]:
        print(f"  ⚠  '{custom_id}' — missing: {parsed['missing_sections']}")
    if final_grade is None:
        print(f"  ⚠  '{custom_id}' — could not compute final grade.")

    row = {
        "run_id":               meta["run_id"],
        "student_id":           meta["student_id"],
        "pipeline_id":          meta["pipeline_id"],
        "model":                meta["model"],
        "temperature":          meta["temperature"],
        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_tokens":         input_tokens,
        "output_tokens":        output_tokens,
        "intro_grade":          parsed["intro_grade"],
        "methods_grade":        parsed["methods_grade"],
        "results_grade":        parsed["results_grade"],
        "discussion_grade":     parsed["discussion_grade"],
        "lang_style_grade":     parsed["lang_style_grade"],
        "final_grade":          final_grade,
        "missing_sections":     parsed["missing_sections"],
        "intro_feedback":       parsed["intro_feedback"],
        "methods_feedback":     parsed["methods_feedback"],
        "results_feedback":     parsed["results_feedback"],
        "discussion_feedback":  parsed["discussion_feedback"],
        "lang_style_feedback":  parsed["lang_style_feedback"],
    }
    for k in CHECKLIST_KEYS:
        row[k] = parsed[k]

    return row


# Main execution

if __name__ == "__main__":

    start_time = time.time()

    if not os.path.exists(BATCH_JSONL):
        print(f"  ✗ JSONL file not found: {BATCH_JSONL}")
        exit(1)

    print(f"Loading metadata from {BATCH_JSONL}...")
    metadata = load_metadata_from_jsonl(BATCH_JSONL)

    # Resume support
    completed: set[str] = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, encoding="utf-8") as f:
            completed = {row["run_id"] for row in csv.DictReader(f) if "run_id" in row}
    print(f"  Skipping {len(completed)} already completed run(s).")

    # Filter out completed requests
    pending_lines = []
    with open(BATCH_JSONL, encoding="utf-8") as f:
        for line in f:
            req = json.loads(line)
            if req["custom_id"] not in completed:
                pending_lines.append(line.strip())

    print(f"  Requests to submit: {len(pending_lines)}\n")
    if not pending_lines:
        print("Nothing to submit. Exiting.")
        exit(0)

    # Upload
    print("Uploading batch input file...")
    jsonl_bytes = "\n".join(pending_lines).encode("utf-8")
    batch_file  = client.files.create(
        file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
        purpose="batch",
    )
    print(f"  ✓ Uploaded — file_id: {batch_file.id}")

    # Submit
    print("Submitting batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    print(f"  ✓ Submitted — batch_id: {batch_job.id}  |  status: {batch_job.status}\n")

    # Poll completion
    print(f"Polling for completion...\n{'─' * 64}")
    while True:
        time.sleep(POLL_INTERVAL)
        batch_job = client.batches.retrieve(batch_job.id)
        elapsed   = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        c         = batch_job.request_counts
        print(f"  [{elapsed}]  status={batch_job.status}  |  total={c.total}  completed={c.completed}  failed={c.failed}")
        if batch_job.status in ("completed", "failed", "expired", "cancelled"):
            break

    elapsed_final = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"{'─' * 64}\n  Batch finished — status: {batch_job.status}  |  elapsed: {elapsed_final}\n")

    if batch_job.status != "completed":
        print(f"  ✗ Batch did not complete (status: {batch_job.status}). Check dashboard for {batch_job.id}")
        exit(1)

    # Download & parse results
    print("Downloading results...")
    result_lines = [json.loads(l) for l in client.files.content(batch_job.output_file_id).text.strip().splitlines()]
    print(f"  ✓ Retrieved {len(result_lines)} result line(s).\n")

    print("Parsing and saving results...")
    saved = errors = total_input_tokens = total_output_tokens = 0

    for line in result_lines:
        row = process_result(line, metadata)
        if row is not None:
            append_result_row(RESULTS_CSV, row)
            saved += 1
            total_input_tokens  += row["input_tokens"]  or 0
            total_output_tokens += row["output_tokens"] or 0
        else:
            errors += 1

    print(f"\n{'=' * 64}")
    print(f"  Batch grading complete.")
    print(f"  Saved          : {saved} row(s)")
    print(f"  Errors         : {errors} row(s)")
    if errors:
        print(f"  ⚠  Check error messages above for details.")
    print(f"  Input tokens   : {total_input_tokens:,}")
    print(f"  Output tokens  : {total_output_tokens:,}")
    print(f"  Output         : {RESULTS_CSV}")
    print(f"  Elapsed        : {elapsed_final}")
    print(f"{'=' * 64}")