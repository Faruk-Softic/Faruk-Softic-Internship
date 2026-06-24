import os
import sys
import json
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import (
    SECTION_KEYS, SECTION_LABELS, CHECKLIST,
    get_output_schema, get_latest_run_folder,
)

load_dotenv()

# Path definitions

PAPERS_FOLDER      = Path(os.environ["PAPERS_FOLDER"])
GRADING_MATERIALS  = Path(os.environ["GRADING_MATERIALS_FOLDER"])
EXPERIMENTS_FOLDER = Path(os.environ["EXPERIMENTS_FOLDER"])
PIPELINES_FILE     = Path(os.environ["PIPELINES_FILE"])

RUBRIC_FILES = {
    "original": GRADING_MATERIALS / os.environ["RUBRIC_ORIGINAL"],
    "improved":  GRADING_MATERIALS / os.environ["RUBRIC_IMPROVED"],
}
RESOURCE_FILES = {
    "writing_guide":  GRADING_MATERIALS / os.environ["WRITING_GUIDE"],
    "sample_results": GRADING_MATERIALS / os.environ["SAMPLE_RESULTS"],
}

EXAMPLE_PAPER_ID = os.environ.get("EXAMPLE_PAPER_ID", "").strip()

# File loading

def load_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        import docx
        from docx.oxml.ns import qn
        from docx.table import Table
        doc   = docx.Document(path)
        parts = []
        for block in doc.element.body:
            tag = block.tag.split("}")[-1]
            if tag == "p":
                text = "".join(node.text or "" for node in block.iter() if node.tag == qn("w:t"))
                if text.strip():
                    parts.append(text)
            elif tag == "tbl":
                for row in Table(block, doc).rows:
                    row_text = "\t".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip():
                        parts.append(row_text)
        return "\n".join(parts)
    if path.suffix.lower() == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return path.read_text(encoding="utf-8")


def load_pipelines(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_papers(folder: Path) -> dict[str, str]:
    papers = {
        p.stem: load_text(p)
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in {".txt", ".md"}
    }
    if not papers:
        raise FileNotFoundError(f"No .txt or .md files found in {folder}")
    return papers

# Prompt building

def build_system_prompt(
    pipeline: dict,
    rubric: str,
    resources: dict[str, str],
    include_feedback: bool,
) -> str:
    include_checklist = pipeline["include_checklist"]
    schema            = get_output_schema(include_checklist, include_feedback)

    # Task instructions
    if include_checklist:
        task_steps = [
            "For each section, you must:",
            "1. Evaluate each checklist item (true/false).",
        ]
        step = 2
        if include_feedback:
            task_steps.append(f"{step}. Write a brief feedback comment.")
            step += 1
        task_steps.append(f"{step}. Assign a numeric grade (1.0–10.0).")
    elif include_feedback:
        task_steps = [
            "For each section, you must:",
            "1. Write a brief feedback comment.",
            "2. Assign a numeric grade (1.0–10.0).",
        ]
    else:
        task_steps = ["For each section, assign a numeric grade (1.0–10.0)."]

    # Output schema block — section headers + keys, no trailing comma on last item
    schema_lines = []
    current_sec  = None
    for i, (key, type_hint) in enumerate(schema):
        sec = key.split("_grade")[0].split("_feedback")[0]
        sec = next((s for s in SECTION_KEYS if key.startswith(s)), current_sec)
        if sec != current_sec:
            current_sec = sec
            schema_lines.append(f"\n  // {SECTION_LABELS[sec]}")
        comma = "," if i < len(schema) - 1 else ""
        schema_lines.append(f'  "{key}": {type_hint}{comma}')

    lines = [
        "You are an expert grader for undergraduate psychology research papers.",
        "You will be given a student paper and grading materials.",
        "",
        "## Writing Guide",
        resources["writing_guide"],
        "",
        "## Sample Results Section",
        resources["sample_results"],
        "",
        "## Grading Rubric",
        rubric,
        "",
        "## Your Task",
        "Grade the paper section by section using the rubric above.",
        "",
        *task_steps,
        "",
        "## Output Format",
        "Respond with a single JSON object. Do not include any text outside the JSON.",
        "Use the exact keys listed below.",
        "",
        "Keys to include:",
        *schema_lines,
    ]
    return "\n".join(lines)


def build_user_prompt(paper_text: str) -> str:
    return f"Please grade the following student paper:\n\n{paper_text}"


def build_request(
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str,
) -> dict:
    return {
        "custom_id": custom_id,
        "method":    "POST",
        "url":       "/v1/responses",
        "body": {
            "model":        model,
            "reasoning":    {"effort": reasoning_effort},
            "text":         {"format": {"type": "json_object"}},
            "instructions": system_prompt,
            "input":        user_prompt,
        },
    }


def make_custom_id(
    run_label: str,
    student_id: str,
    pipeline_id: str,
    run_n: int,
    total_runs: int,
) -> str:
    base = f"{run_label}__{student_id}__{pipeline_id}"
    return f"{base}__run{run_n}" if total_runs > 1 else base

# Main

if __name__ == "__main__":
    run_folder = get_latest_run_folder(EXPERIMENTS_FOLDER)
    run_label  = run_folder.name
    print(f"Using run folder: {run_folder}")

    cfg = yaml.safe_load((run_folder / "config.yml").read_text(encoding="utf-8"))

    model            = cfg.get("model", "gpt-4.5")
    reasoning_effort = cfg.get("reasoning_effort", "medium")
    include_feedback = cfg.get("include_feedback", False)
    repetitions      = cfg.get("repetitions", 1)
    requested_pids   = cfg.get("pipelines") or None        # None = all
    student_ids_cfg  = cfg.get("student_ids") or []        # [] = all

    print(f"  model={model}, reasoning_effort={reasoning_effort}, "
          f"include_feedback={include_feedback}, repetitions={repetitions}")

    print("Loading rubrics...")
    rubrics = {}
    for name, path in RUBRIC_FILES.items():
        rubrics[name] = load_text(path)
        print(f"  ✓ '{name}' loaded.")

    print("Loading resources...")
    resources = {name: load_text(path) for name, path in RESOURCE_FILES.items()}
    for name in resources:
        print(f"  ✓ '{name}' loaded.")

    print("Loading pipelines...")
    all_pipelines = load_pipelines(PIPELINES_FILE)
    pipelines = (
        [all_pipelines[pid] for pid in requested_pids if pid in all_pipelines]
        if requested_pids else list(all_pipelines.values())
    )
    for p in pipelines:
        print(f"  ✓ '{p['pipeline_id']}' loaded.")

    print("Loading papers...")
    all_papers = load_papers(PAPERS_FOLDER)
    papers = (
        {sid: all_papers[sid] for sid in student_ids_cfg if sid in all_papers}
        if student_ids_cfg else all_papers
    )
    print(f"  ✓ {len(papers)} paper(s) found.")

    # Build requests — in the order: paper - pipeline - repetition
    requests     = []
    example_reqs = []
    example_id   = EXAMPLE_PAPER_ID or next(iter(papers))

    for student_id, paper_text in papers.items():
        for pipeline in pipelines:
            pid    = pipeline["pipeline_id"]
            rubric = rubrics[pipeline["rubric_version"]]
            sys_p  = build_system_prompt(pipeline, rubric, resources, include_feedback)
            user_p = build_user_prompt(paper_text)

            for run_n in range(1, repetitions + 1):
                custom_id = make_custom_id(run_label, student_id, pid, run_n, repetitions)
                req       = build_request(custom_id, sys_p, user_p, model, reasoning_effort)
                requests.append(req)
                if student_id == example_id and run_n == 1:
                    example_reqs.append(req)

    # Write batch input JSONL
    batch_path = run_folder / "batch_input.jsonl"
    with open(batch_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    print(f"  ✓ batch_input.jsonl written ({len(requests)} requests).")

    # Write example JSON
    example_path = run_folder / "prompts_example.json"
    example_path.write_text(
        json.dumps(example_reqs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  ✓ prompts_example.json written (paper: '{example_id}').")

    # Write manifest
    manifest = {
        "run_id":           run_label,
        "created_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":            model,
        "reasoning_effort": reasoning_effort,
        "include_feedback": include_feedback,
        "repetitions":      repetitions,
        "pipelines":        [p["pipeline_id"] for p in pipelines],
        "n_papers":         len(papers),
        "n_requests":       len(requests),
        "status":           "pending",
        "notes":            cfg.get("notes", ""),
    }
    manifest_path = run_folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  ✓ manifest.json written.")

    print(f"\nDone. Review '{run_folder}' before submitting.")