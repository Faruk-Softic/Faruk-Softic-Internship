"""
Submits the batch_input.jsonl from the latest run folder,
polls for completion, downloads output, parses results to CSV, and
finalises manifest.json.

To cancel mid-run, press Ctrl+C.
"""

import os
import sys
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import (
    SECTION_KEYS, CHECKLIST_KEYS, SECTION_WEIGHTS,
    get_output_schema, get_csv_fieldnames, get_latest_run_folder,
)

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EXPERIMENTS_FOLDER = Path(os.environ["EXPERIMENTS_FOLDER"])
OUTPUTS_FOLDER     = Path(os.environ["OUTPUTS_FOLDER"])

# JSON/manifest helpers

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

# Parsing helpers

def _parse_grade(value) -> Optional[float]:
    try:
        g = float(value)
        return round(g * 2) / 2.0 if 1.0 <= g <= 10.0 else None
    except (TypeError, ValueError):
        return None


def _get_tokens(result_line: dict) -> tuple[Optional[int], Optional[int]]:
    usage = result_line.get("response", {}).get("body", {}).get("usage", {})
    return usage.get("prompt_tokens"), usage.get("completion_tokens")


def _extract_reply(result_line: dict) -> Optional[str]:
    try:
        return result_line["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def parse_reply(raw: str, include_checklist: bool, include_feedback: bool) -> dict:
    schema = get_output_schema(include_checklist, include_feedback)
    base   = {key: None for key, _ in schema}
    base["raw_reply"]        = raw
    base["missing_sections"] = ""

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return base

    for key, _ in schema:
        if key.endswith("_grade"):
            base[key] = _parse_grade(data.get(key))
        elif key.endswith("_feedback"):
            base[key] = data.get(key)
        else:
            val = data.get(key)
            base[key] = bool(val) if val is not None else None

    missing = [k for k in SECTION_KEYS if base.get(f"{k}_grade") is None]
    base["missing_sections"] = ", ".join(missing)
    return base


def load_metadata(jsonl_path: Path) -> dict:
    metadata = {}
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        req         = json.loads(line)
        custom_id   = req["custom_id"]
        parts       = custom_id.split("__")
        metadata[custom_id] = {
            "run_id":           custom_id,
            "run_label":        parts[0],
            "student_id":       parts[1],
            "pipeline_id":      parts[2],
            "repetition":       parts[3] if len(parts) > 3 else "run1",
            "model":            req["body"]["model"],
            "reasoning_effort": req["body"].get("reasoning_effort"),
        }
    return metadata


def write_result_row(csv_path: Path, row: dict) -> None:
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
    include_checklist: bool,
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

    raw_reply = _extract_reply(result_line)
    if raw_reply is None:
        print(f"  ✗  Could not extract reply for '{custom_id}'.")
        return None

    input_tokens, output_tokens = _get_tokens(result_line)
    parsed     = parse_reply(raw_reply, include_checklist, include_feedback)
    grade_vals = {k: parsed.get(f"{k}_grade") for k in SECTION_KEYS}
    final_grade = (
        sum(grade_vals[k] * SECTION_WEIGHTS[k] for k in SECTION_KEYS)
        if all(v is not None for v in grade_vals.values()) else None
    )

    if parsed["missing_sections"]:
        print(f"  ⚠  '{custom_id}' — missing grades: {parsed['missing_sections']}")
    if final_grade is None:
        print(f"  ⚠  '{custom_id}' — could not compute final grade.")

    # Build row using centralized fieldnames to guarantee column order
    fieldnames = get_csv_fieldnames(include_checklist, include_feedback)
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
        **{f"{k}_grade": parsed.get(f"{k}_grade") for k in SECTION_KEYS},
        "final_grade":      round(final_grade, 2) if final_grade is not None else None,
        "missing_sections": parsed["missing_sections"],
        **({f"{k}_feedback": parsed.get(f"{k}_feedback") for k in SECTION_KEYS} if include_feedback else {}),
        **({k: parsed.get(k) for k in CHECKLIST_KEYS} if include_checklist else {}),
    }
    # Ensure only declared fieldnames are written (guards against schema drift)
    return {k: row.get(k) for k in fieldnames}

# Main

if __name__ == "__main__":
    run_folder    = get_latest_run_folder(EXPERIMENTS_FOLDER)
    run_label     = run_folder.name
    batch_path    = run_folder / "batch_input.jsonl"
    manifest_path = run_folder / "manifest.json"

    manifest          = load_json(manifest_path)
    include_feedback  = manifest.get("include_feedback", False)
    # Infer include_checklist from which pipelines are in this run
    pipeline_ids      = manifest.get("pipelines", [])
    include_checklist = "improved" in pipeline_ids

    print(f"Submitting '{run_label}' from '{batch_path}'...")

    with open(batch_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"  ✓ File uploaded: {uploaded.id}")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  ✓ Batch submitted: {batch.id}")

    manifest.update({
        "batch_id":     batch.id,
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status":       "submitted",
    })
    save_json(manifest_path, manifest)

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
        save_json(manifest_path, manifest)
        print(f"  ✓ Batch {batch.id} cancelled. Manifest updated.")
        raise SystemExit(0)

    if batch.status != "completed":
        manifest["status"] = batch.status
        save_json(manifest_path, manifest)
        print(f"  ✗ Batch ended with status '{batch.status}'. Exiting.")
        raise SystemExit(1)

    # Download output JSONL
    output_jsonl_path = run_folder / f"{run_label}_output.jsonl"
    output_jsonl_path.write_bytes(client.files.content(batch.output_file_id).content)
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
        row = process_result(line, metadata, include_checklist, include_feedback)
        if row is not None:
            write_result_row(csv_path, row)
            write_result_row(out_csv, row)
            saved     += 1
            total_in  += row.get("input_tokens")  or 0
            total_out += row.get("output_tokens") or 0
        else:
            errors += 1

    manifest.update({
        "status":              "completed",
        "completed_at":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_input_tokens":  total_in,
        "total_output_tokens": total_out,
        "n_saved":             saved,
        "n_errors":            errors,
    })
    save_json(manifest_path, manifest)
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
