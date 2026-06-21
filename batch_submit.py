"""
This script submits the batch_input.jsonl from the latest experiments/run-XX folder,
polls for completion, downloads output, parses results to CSV, and
finalises manifest.json.

Press Ctrl+C during polling to cancel the batch run.
"""

import os
import sys
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

# Allow imports from /src
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import SECTION_KEYS, CHECKLIST_KEYS, SECTION_WEIGHTS

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EXPERIMENTS_FOLDER = Path(os.environ["EXPERIMENTS_FOLDER"])
OUTPUTS_FOLDER     = Path(os.environ["OUTPUTS_FOLDER"])

# Run folder detection
def get_latest_run_folder(base: Path) -> Path:
    folders = [
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r"run-\d+", d.name)
    ]
    if not folders:
        raise FileNotFoundError(f"No run folders found in '{base}'.")
    return max(folders, key=lambda d: int(d.name.split("-")[1]))

# Parsing helpers
def _parse_grade(value) -> Optional[float]:
    try:
        g = float(value)
        return round(g * 2) / 2.0 if 1.0 <= g <= 10.0 else None
    except (TypeError, ValueError):
        return None


def parse_reply(raw: str, include_feedback: bool) -> dict:
    base = {f"{k}_grade": None for k in SECTION_KEYS}
    base.update({f"{k}_feedback": None for k in SECTION_KEYS})
    base.update({k: None for k in CHECKLIST_KEYS})
    base["raw_reply"]        = raw
    base["missing_sections"] = ""

    try:
        data = json.loads(raw)
    except Exception:
        return base

    for k in SECTION_KEYS:
        base[f"{k}_grade"] = _parse_grade(data.get(f"{k}_grade"))
        if include_feedback:
            base[f"{k}_feedback"] = data.get(f"{k}_feedback")

    for k in CHECKLIST_KEYS:
        val = data.get(k)
        base[k] = bool(val) if val is not None else None

    missing = [k for k in SECTION_KEYS if base[f"{k}_grade"] is None]
    base["missing_sections"] = ", ".join(missing)
    return base


def load_metadata(jsonl_path: Path) -> dict:
    metadata = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            req         = json.loads(line)
            custom_id   = req["custom_id"]
            # Format: run-XX__studentID__pipelineID__repN
            parts       = custom_id.split("__")
            run_label   = parts[0]
            student_id  = parts[1]
            pipeline_id = parts[2]
            repetition  = parts[3] if len(parts) > 3 else "rep1"
            metadata[custom_id] = {
                "run_id":           custom_id,
                "run_label":        run_label,
                "student_id":       student_id,
                "pipeline_id":      pipeline_id,
                "repetition":       repetition,
                "model":            req["body"]["model"],
                "reasoning_effort": req["body"].get("reasoning_effort"),
            }
    return metadata


def append_result_row(csv_path: Path, row: dict) -> None:
    file_exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def process_result(
    result_line: dict,
    metadata: dict,
    include_feedback: bool,
) -> Optional[dict]:
    custom_id = result_line.get("custom_id", "unknown")
    meta      = metadata.get(custom_id)

    if meta is None:
        print(f"  ⚠  No metadata for '{custom_id}' — skipping.")
        return None
    if error := result_line.get("error"):
        print(f"  ✗  API error for '{custom_id}': {error}")
        return None
    try:
        raw_reply = result_line["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        print(f"  ✗  Could not extract reply for '{custom_id}': {e}")
        return None

    usage         = result_line.get("response", {}).get("body", {}).get("usage", {})
    input_tokens  = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    parsed     = parse_reply(raw_reply, include_feedback)
    grade_vals = {k: parsed[f"{k}_grade"] for k in SECTION_KEYS}
    final_grade = (
        sum(grade_vals[k] * SECTION_WEIGHTS[k] for k in SECTION_KEYS)
        if all(v is not None for v in grade_vals.values()) else None
    )

    if parsed["missing_sections"]:
        print(f"  ⚠  '{custom_id}' — missing grades: {parsed['missing_sections']}")
    if final_grade is None:
        print(f"  ⚠  '{custom_id}' — could not compute final grade.")

    row = {
        "run_id":           meta["run_id"],
        "run_label":        meta["run_label"],
        "student_id":       meta["student_id"],
        "pipeline_id":      meta["pipeline_id"],
        "repetition":       meta["repetition"],
        "model":            meta["model"],
        "reasoning_effort": meta["reasoning_effort"],
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_tokens":     input_tokens,
        "output_tokens":    output_tokens,
        "intro_grade":      parsed["intro_grade"],
        "methods_grade":    parsed["methods_grade"],
        "results_grade":    parsed["results_grade"],
        "discussion_grade": parsed["discussion_grade"],
        "lang_style_grade": parsed["lang_style_grade"],
        "final_grade":      round(final_grade, 2) if final_grade is not None else None,
        "missing_sections": parsed["missing_sections"],
    }

    if include_feedback:
        for k in SECTION_KEYS:
            row[f"{k}_feedback"] = parsed[f"{k}_feedback"]

    for k in CHECKLIST_KEYS:
        row[k] = parsed[k]

    return row


# Manifest helpers
def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


# Main
if __name__ == "__main__":
    run_folder    = get_latest_run_folder(EXPERIMENTS_FOLDER)
    run_label     = run_folder.name
    batch_path    = run_folder / "batch_input.jsonl"
    manifest_path = run_folder / "manifest.json"

    # Load manifest to get include_feedback setting
    manifest         = load_manifest(manifest_path)
    include_feedback = manifest.get("include_feedback", False)

    print(f"Submitting '{run_label}' from '{batch_path}'...")

    # Upload file
    with open(batch_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  ✓ File uploaded: {uploaded.id}")

    # Submit batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  ✓ Batch submitted: {batch.id}")

    # Update manifest with batch ID and submission time
    manifest["batch_id"]     = batch.id
    manifest["submitted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    manifest["status"]       = "submitted"
    save_manifest(manifest_path, manifest)

    
    print("  Polling for completion (Ctrl+C to cancel)...", end="", flush=True)
    try:
        while batch.status not in {"completed", "failed", "expired", "cancelled"}:
            time.sleep(30)
            batch = client.batches.retrieve(batch.id)
            print(".", end="", flush=True)
        print(f" {batch.status.upper()}")
    except KeyboardInterrupt:
        print("\n  Cancelling batch on OpenAI...")
        client.batches.cancel(batch.id)
        manifest["status"] = "cancelled"
        save_manifest(manifest_path, manifest)
        print(f"  ✓ Batch {batch.id} cancelled. Manifest updated.")
        exit(0)

    if batch.status != "completed":
        manifest["status"] = batch.status
        save_manifest(manifest_path, manifest)
        print(f"  ✗ Batch ended with status '{batch.status}'. Exiting.")
        exit(1)

    # Download output JSONL
    output_jsonl_path = run_folder / f"{run_label}_output.jsonl"
    content = client.files.content(batch.output_file_id).content
    with open(output_jsonl_path, "wb") as f:
        f.write(content)
    print(f"  ✓ Output JSONL saved: '{output_jsonl_path}'")

    # Parse results
    result_lines = [
        json.loads(line)
        for line in output_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    metadata = load_metadata(batch_path)

    csv_path = run_folder / f"{run_label}_results_raw.csv"
    out_csv  = OUTPUTS_FOLDER / run_label / f"{run_label}_results_raw.csv"

    print(f"\nParsing {len(result_lines)} result(s)...")
    saved = errors = total_in = total_out = 0

    for line in result_lines:
        row = process_result(line, metadata, include_feedback)
        if row is not None:
            append_result_row(csv_path, row)
            append_result_row(out_csv, row)
            saved     += 1
            total_in  += row["input_tokens"]  or 0
            total_out += row["output_tokens"] or 0
        else:
            errors += 1

    # Finalise manifest
    manifest.update({
        "status":              "completed",
        "completed_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_input_tokens":  total_in,
        "total_output_tokens": total_out,
        "n_saved":             saved,
        "n_errors":            errors,
    })
    save_manifest(manifest_path, manifest)
    print(f"  ✓ manifest.json finalised.")

    print(f"\n{'=' * 64}")
    print(f"  Run            : {run_label}")
    print(f"  Saved          : {saved} row(s)")
    print(f"  Errors         : {errors} row(s)")
    print(f"  Input tokens   : {total_in:,}")
    print(f"  Output tokens  : {total_out:,}")
    print(f"  Experiments    : {run_folder}")
    print(f"  Outputs        : {out_csv.parent}")
    print(f"{'=' * 64}")
