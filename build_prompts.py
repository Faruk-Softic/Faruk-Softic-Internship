"""

Builds batch_input.jsonl and prompts_example.json for the latest run folder.

"""

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


_GENERAL_INSTRUCTIONS = (
    "# General instructions\n"
    "You are grading a paper written by a second-year psychology bachelor student as part of a writing course "
    "focused on scientific reasoning and argumentation. Grade the paper according to the expectations of this "
    "specific course, not according to the standards of publishable scientific articles or graduate-level writing. "
    "Base every judgment on evidence from the student's paper and the provided grading materials. "
    "If an abstract is included, ignore it as abstracts are not graded in this course. "
    "Assign grades from 1.0 to 10.0 in 0.5-point increments according to the Dutch grading system: "
    "grades between 1-5 are failing; 5.5 is the lowest passing grade; grades between 6.5-7.5 are average; "
    "grades between 7.5-9 are above average; grades between 9-10 are excellent. "
    "Be as accurate and meticulous as possible."
)

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

_SEC_PREFIX = {
    "intro":      "intro",
    "methods":    "methods",
    "results":    "results",
    "discussion": "discussion",
    "lang_style": "lang",
}


def build_system_prompt(
    pipeline: dict,
    rubric: str,
    resources: dict[str, str],
    include_feedback: bool,
) -> str:
    """
    Build the full system prompt for a given pipeline.

    Output order per section:
      1. Checklist items  (true/false) — only for pipelines with include_checklist: true
      2. Feedback comment             — only if include_feedback=True
      3. Numeric grade
    """
    include_checklist = pipeline.get("include_checklist", False)
    schema            = get_output_schema(include_checklist, include_feedback)

    # Task instruction block
    if include_checklist:
        task_steps = [
            "For each section, work through the following steps in order:",
            "1. Evaluate each checklist item (true/false) based on the rubric criteria.",
        ]
        step = 2
        if include_feedback:
            task_steps.append(f"{step}. Write a feedback comment explaining your reasoning behind the grade.")
            step += 1
        task_steps.append(
            f"{step}. Assign a numeric grade (1.0-10.0 in 0.5 steps). "
            "The grade must be consistent with the tier reached based on the checklist: "
            "a paper that fails to meet all core criteria for a tier cannot be graded within "
            "that tier's range, regardless of additional criteria."
        )
    elif include_feedback:
        task_steps = [
            "For each section, work through the following steps in order:",
            "1. Write a feedback comment explaining your reasoning behind the grade.",
            "2. Assign a numeric grade (1.0-10.0 in 0.5 steps).",
        ]
    else:
        task_steps = ["For each section, assign a numeric grade (1.0-10.0 in 0.5 steps)."]

    # Output schema block
    schema_lines = []
    current_sec  = None
    for i, (key, type_hint) in enumerate(schema):
        sec = next(
            (s for s in SECTION_KEYS if key.startswith(_SEC_PREFIX[s])),
            current_sec,
        )
        if sec != current_sec:
            current_sec = sec
            schema_lines.append(f"\n  // {SECTION_LABELS[sec]}")
        comma = "," if i < len(schema) - 1 else ""
        schema_lines.append(f'  "{key}": {type_hint}{comma}')

    lines = [
        _GENERAL_INSTRUCTIONS,
        "",
        "# Writing guide",
        "Students were instructed to follow this writing guide.",
        "[WRITING GUIDE START]",
        resources["writing_guide"],
        "[WRITING GUIDE END]",
        "",
        "# Sample results section",
        "Students used simulated data and were encouraged to follow the overall structure, reporting style, "
        "and level of detail illustrated in the sample results section below. However, the sample is not an "
        "answer key. Do not reward or penalize deviations from the sample unless they omit required information "
        "or violate the writing guide or grading rubric.",
        "[SAMPLE RESULTS START]",
        resources["sample_results"],
        "[SAMPLE RESULTS END]",
        "",
        "# Grading rubric",
        "Use the grading rubric as the primary grading standard. Use the writing guide and sample results "
        "section to interpret the expectations of the assignment. Assign grades from 1.0 to 10.0 in 0.5-point "
        "increments according to the rubric criteria below.",
        "[RUBRIC START]",
        rubric,
        "[RUBRIC END]",
        "",
        "# Response format",
        *task_steps,
        "",
        "Respond with a single JSON object. Do not include any text outside the JSON.",
        "Use the exact keys listed below, in the order shown.",
        "",
        "{",
        *schema_lines,
        "}",
    ]
    return "\n".join(lines)


def build_user_prompt(paper_text: str) -> str:
    return "\n".join([
        "# Student paper",
        "Grade the following student paper:",
        "[STUDENT PAPER START]",
        paper_text,
        "[STUDENT PAPER END]",
        "",
        "Respond with a JSON object only.",
    ])


def build_request(
    custom_id: str,
    system_prompt: str,
    user_prompt: str,
    model: str,
    reasoning_effort: str,
    reasoning_summary: str,
) -> dict:
    return {
        "custom_id": custom_id,
        "method":    "POST",
        "url":       "/v1/responses",
        "body": {
            "model":        model,
            "reasoning":    {"effort": reasoning_effort, "summary": reasoning_summary},
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

    model             = cfg["model"]
    reasoning_effort  = cfg.get("reasoning_effort", "medium")
    reasoning_summary = cfg.get("reasoning_summary", "auto")
    include_feedback  = cfg.get("include_feedback", False)
    repetitions       = cfg.get("repetitions", 1)
    requested_pids    = cfg.get("pipelines") or None   # None = all
    student_ids_cfg   = cfg.get("student_ids") or []   # [] = all

    print(
        f"  model={model}, reasoning_effort={reasoning_effort}, "
        f"reasoning_summary={reasoning_summary}, "
        f"include_feedback={include_feedback}, "
        f"repetitions={repetitions}"
    )

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
        print(f"  ✓ '{p['pipeline_id']}' loaded "
              f"(rubric={p['rubric_version']}, "
              f"checklist={p.get('include_checklist', False)}).")

    print("Loading papers...")
    all_papers = load_papers(PAPERS_FOLDER)
    papers = (
        {sid: all_papers[sid] for sid in student_ids_cfg if sid in all_papers}
        if student_ids_cfg else all_papers
    )
    print(f"  ✓ {len(papers)} paper(s) found.")

    # Build requests: paper, then pipeline, then repetition
    requests     = []
    example_reqs = []
    example_id   = EXAMPLE_PAPER_ID or next(iter(papers))

    for student_id, paper_text in papers.items():
        for pipeline in pipelines:
            pid    = pipeline["pipeline_id"]
            rubric = rubrics[pipeline["rubric_version"]]
            sys_p  = build_system_prompt(
                pipeline, rubric, resources, include_feedback
            )
            user_p = build_user_prompt(paper_text)

            for run_n in range(1, repetitions + 1):
                custom_id = make_custom_id(run_label, student_id, pid, run_n, repetitions)
                req       = build_request(
                    custom_id, sys_p, user_p, model, reasoning_effort, reasoning_summary
                )
                requests.append(req)
                if student_id == example_id and run_n == 1:
                    example_reqs.append(req)

    # Write batch_input.jsonl
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
        "run_id":             run_label,
        "created_at":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":              model,
        "reasoning_effort":   reasoning_effort,
        "reasoning_summary":  reasoning_summary,
        "include_feedback":   include_feedback,
        "repetitions":        repetitions,
        "pipelines":          [p["pipeline_id"] for p in pipelines],
        "n_papers":           len(papers),
        "n_requests":         len(requests),
        "status":             "pending",
        "notes":              cfg.get("notes", ""),
    }
    manifest_path = run_folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  ✓ manifest.json written.")

    print(f"\nDone. Review '{run_folder}' before submitting.")