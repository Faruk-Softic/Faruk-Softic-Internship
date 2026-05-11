"""
This script grades papers in one go, without intermediate grades.
"""

import os
import csv
import json
import glob
import re
from datetime import datetime
from typing import Optional

import pdfplumber
import tiktoken
from docx import Document
from openai import OpenAI


# Configure paths here, if necessary

API_KEY_FILE         = r"C:\Users\fsoft\Desktop\api_key.txt"
PAPERS_FOLDER        = r"C:\Users\fsoft\Desktop\Za Internship - code\Papers"
RUBRIC_ORIGINAL_PATH = r"C:\Users\fsoft\Desktop\Za Internship - code\Rubric_original.docx"
RUBRIC_IMPROVED_PATH = r"C:\Users\fsoft\Desktop\Za Internship - code\Rubric_improved.docx"
PIPELINES_FILE       = r"C:\Users\fsoft\Desktop\Za Internship - code\pipelines.json"
RESULTS_CSV          = r"C:\Users\fsoft\Desktop\Za Internship - code\results.csv"

# If needed, change number of runs per pipeline

N_RUNS = 5

SECTION_ORDER = ["introduction", "methods", "results", "discussion"]

IMPROVED_WEIGHTS = {
    "introduction":   0.285,
    "methods":        0.190,
    "results":        0.190,
    "discussion":     0.285,
    "language_style": 0.050,
}

# Role description used in every prompt
LLM_ROLE = (
    "You are a tutor for a course designed to teach second-year psychology students "
    "how to write scientific papers, with a particular focus on scientific reasoning "
    "and argumentation."
)


# API key part

def _load_api_key() -> str:
    if not os.path.exists(API_KEY_FILE):
        raise FileNotFoundError(
            f"API key file not found: {API_KEY_FILE}\n"
            "API Key file missing."
        )
    with open(API_KEY_FILE, "r", encoding="utf-8") as f:
        key = f.readline().strip()
    if not key:
        raise ValueError(f"API key file is empty: {API_KEY_FILE}")
    return key


client = OpenAI(api_key=_load_api_key())


# This part defines how papers will be parsed for both docx and pdf file types

def _read_pdf(path: str) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text)
            for table in page.extract_tables() or []:
                for row in table:
                    row_text = " | ".join(
                        cell.strip() for cell in row if cell and cell.strip()
                    )
                    if row_text:
                        parts.append(row_text)
    return "\n\n".join(parts)


def _read_docx(path: str) -> str:
    doc = Document(path)
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)
    return "\n".join(parts)


def _split_into_sections(text: str) -> dict[str, str]:
    """
    Splits document text into four sections as defined by subheading titles, accounting for slight name variation
    """
    section_patterns = {
        "introduction": r"(?i)^\s*(\d+[\.\)]\s*)?introduction[:\.]?\s*$",
        "methods":      r"(?i)^\s*(\d+[\.\)]\s*)?methods?[:\.]?\s*$",
        "results":      r"(?i)^\s*(\d+[\.\)]\s*)?results?[:\.]?\s*$",
        "discussion":   r"(?i)^\s*(\d+[\.\)]\s*)?discussion[:\.]?\s*$",
    }

    section_texts: dict[str, str] = {}
    lines = text.split("\n")
    current_section: Optional[str] = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if len(stripped) <= 40:
            matched = False
            for sec_name, pattern in section_patterns.items():
                if re.match(pattern, stripped):
                    if current_section is not None:
                        section_texts[current_section] = "\n".join(current_lines).strip()
                    current_section = sec_name
                    current_lines = []
                    matched = True
                    break
            if matched:
                continue
        if current_section is not None:
            current_lines.append(line)

    if current_section is not None:
        section_texts[current_section] = "\n".join(current_lines).strip()

    return section_texts


def load_document(path: str) -> dict[str, str]:
    """Load a .pdf or .docx paper and return {section_name: section_text}."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".pdf":
        full_text = _read_pdf(path)
    elif ext == ".docx":
        full_text = _read_docx(path)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .pdf or .docx.")
    return _split_into_sections(full_text)


def build_full_paper_text(sections: dict[str, str]) -> str:
    """
    If the pipeline is holistic, this functions combines all detected sections into one string for the LLM to process.
    """
    parts = []
    for sec in SECTION_ORDER:
        if sec in sections and sections[sec].strip():
            parts.append(f"## {sec.upper()}\n\n{sections[sec]}")
    return "\n\n---\n\n".join(parts)


# This part checks token length and provides a warning if it's getting close to the limit

_ENCODING   = tiktoken.get_encoding("o200k_base")
_MAX_TOKENS = 100_000


def check_token_count(text: str, label: str = "") -> int:
    n = len(_ENCODING.encode(text))
    tag = f"[{label}] " if label else ""
    if n > _MAX_TOKENS:
        raise ValueError(f"{tag}Text is {n} tokens, exceeding the {_MAX_TOKENS}-token limit.")
    if n > _MAX_TOKENS * 0.85:
        print(f"  ⚠  Warning: {tag}{n} tokens (>85 % of limit).")
    return n


# This part loads both original and improved rubric files, and keeps them in cached memory

def load_rubrics() -> dict[str, str]:
    """Read both rubric files once and keep them in memory."""
    print("\nLoading rubrics into memory cache...")
    rubrics: dict[str, str] = {}
    for name, path in [("original", RUBRIC_ORIGINAL_PATH),
                        ("improved", RUBRIC_IMPROVED_PATH)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Rubric file not found: {path}")
        rubrics[name] = _read_docx(path)
        n = check_token_count(rubrics[name], label=f"rubric_{name}")
        print(f"  ✓ Rubric '{name}' loaded ({n} tokens).")
    return rubrics


# This part loads individual papers and pipelines

def load_pipelines(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        all_pipelines = json.load(f)
    holistic = [p for p in all_pipelines if p.get("grading_mode") == "holistic"]
    print(f"Loaded {len(holistic)} holistic pipeline(s) from {len(all_pipelines)} total.")
    return holistic


def list_papers(folder: str) -> list[tuple[str, str]]:
    """Return sorted (student_id, path) pairs for all papers in folder."""
    paths = (
        glob.glob(os.path.join(folder, "*.pdf")) +
        glob.glob(os.path.join(folder, "*.docx"))
    )
    papers = []
    for p in paths:
        filename = os.path.basename(p)
        if filename.startswith("~$"):
            continue
        student_id = os.path.splitext(filename)[0]
        papers.append((student_id, p))
    return sorted(papers, key=lambda x: x[0])


# This part defines grade parsing and validation (REMEMBER TO EDIT THIS TO REVISE THE GRADE SO THAT EVERYTHING UNDER A 6.0 IS TURNED INTO A FAIL)

def _parse_grade(value) -> Optional[float]:
    """
    Validate and normalise to the Dutch grading scale:
      - Range 1.0 – 10.0, steps of 0.5
      - 5.5 is not valid → rounded down to 5.0
    """
    if value is None:
        return None
    try:
        g = float(value)
    except (TypeError, ValueError):
        return None
    g = round(g * 2) / 2.0
    if g < 1.0 or g > 10.0:
        return None
    if g == 5.5:
        g = 5.0                  
    return g


def parse_holistic_reply_original(raw: str) -> dict:
    """
    Parse the single-call reply for the ORIGINAL rubric.

    Expected JSON keys:
      final_grade, justification,
      intro_rubric_feedback, methods_rubric_feedback,
      results_rubric_feedback, discussion_rubric_feedback,
      lang_style_rubric_feedback
    """
    try:
        data = json.loads(raw)
    except Exception:
        return {
            "final_grade": None, "justification": None,
            "intro_rubric_feedback": None, "methods_rubric_feedback": None,
            "results_rubric_feedback": None, "discussion_rubric_feedback": None,
            "lang_style_rubric_feedback": None,
            "raw_reply": raw,
        }
    return {
        "final_grade":                _parse_grade(data.get("final_grade")),
        "justification":              data.get("justification"),
        "intro_rubric_feedback":      data.get("intro_rubric_feedback"),
        "methods_rubric_feedback":    data.get("methods_rubric_feedback"),
        "results_rubric_feedback":    data.get("results_rubric_feedback"),
        "discussion_rubric_feedback": data.get("discussion_rubric_feedback"),
        "lang_style_rubric_feedback": data.get("lang_style_rubric_feedback"),
        "raw_reply":                  raw,
    }


def parse_holistic_reply_improved(raw: str) -> dict:
    """
    Parse the single-call reply for the IMPROVED rubric.

    Expected JSON keys:
      intro_grade, methods_grade, results_grade, discussion_grade,
      lang_style_grade,
      intro_rubric_feedback, methods_rubric_feedback,
      results_rubric_feedback, discussion_rubric_feedback,
      lang_style_rubric_feedback
    """
    try:
        data = json.loads(raw)
    except Exception:
        return {
            "intro_grade": None, "methods_grade": None,
            "results_grade": None, "discussion_grade": None,
            "lang_style_grade": None,
            "intro_rubric_feedback": None, "methods_rubric_feedback": None,
            "results_rubric_feedback": None, "discussion_rubric_feedback": None,
            "lang_style_rubric_feedback": None,
            "raw_reply": raw,
        }
    return {
        "intro_grade":                _parse_grade(data.get("intro_grade")),
        "methods_grade":              _parse_grade(data.get("methods_grade")),
        "results_grade":              _parse_grade(data.get("results_grade")),
        "discussion_grade":           _parse_grade(data.get("discussion_grade")),
        "lang_style_grade":           _parse_grade(data.get("lang_style_grade")),
        "intro_rubric_feedback":      data.get("intro_rubric_feedback"),
        "methods_rubric_feedback":    data.get("methods_rubric_feedback"),
        "results_rubric_feedback":    data.get("results_rubric_feedback"),
        "discussion_rubric_feedback": data.get("discussion_rubric_feedback"),
        "lang_style_rubric_feedback": data.get("lang_style_rubric_feedback"),
        "raw_reply":                  raw,
    }


# For the improved rubrics, this part defines the weighted formula for calculating final grade

def compute_improved_final_grade(grades: dict[str, Optional[float]]) -> Optional[float]:
    """
    Weighted final grade for the improved rubric.
    Returns None if any required component grade is missing.

    There are some hard limits to the grade as defined in the improved rubrics:
      - Any component < 5.5  → final capped at 6.0
      - Two or more < 5.5    → final capped at 5.0
      - 5.5 is not valid     → rounded down to 5.0
    """
    for key in IMPROVED_WEIGHTS:
        if grades.get(key) is None:
            return None

    weighted = sum(grades[k] * w for k, w in IMPROVED_WEIGHTS.items())
    final = round(weighted * 2) / 2.0

    below = [k for k in IMPROVED_WEIGHTS if grades[k] < 5.5]
    if len(below) >= 2:
        final = min(final, 5.0)
    elif len(below) == 1:
        final = min(final, 6.0)

    if final == 5.5:
        final = 5.0

    return final


# Prompt-building part

def _build_holistic_prompt_original(rubric_text: str) -> str:
    """
    System prompt for the original rubric holistic grading call.

    The model reads the entire paper and assigns:
      - A single holistic final grade + justification
      - Rubric feedback per section + Language & Style
        (about the rubric as a tool, not about the student's paper)
    """
    return f"""{LLM_ROLE}
You will read the entire paper and assign a single holistic final grade.

## Grading instructions
The rubric explicitly states that the final grade is NOT a weighted average of section scores.
It should reflect the paper as a whole, with most weight placed on scientific reasoning and
argumentation — primarily in the Introduction and Discussion.
Use the rubric's performance band descriptions to anchor your judgment.

## Rubric feedback instructions
After assigning the final grade, provide brief feedback on the rubric CRITERIA THEMSELVES
for EACH section and for Language & Style. Imagine you are a rubric designer reviewing your
own tool. Note any criteria that are unclear, overly subjective, or that would benefit from
more concrete anchors or clearer band descriptors.
Do NOT describe or evaluate the student's work in this field.

## Full grading rubric
{rubric_text}

## Output format
Respond with a JSON object containing EXACTLY these keys — no more, no less:
{{
  "final_grade": <number between 1.0 and 10.0, multiples of 0.5, not 5.5>,
  "justification": "<3–5 sentences explaining how the overall paper quality maps onto the rubric bands>",
  "intro_rubric_feedback": "<2–3 sentences of feedback on the Introduction rubric criteria THEMSELVES, not on the student paper>",
  "methods_rubric_feedback": "<2–3 sentences of feedback on the Methods rubric criteria THEMSELVES, not on the student paper>",
  "results_rubric_feedback": "<2–3 sentences of feedback on the Results rubric criteria THEMSELVES, not on the student paper>",
  "discussion_rubric_feedback": "<2–3 sentences of feedback on the Discussion rubric criteria THEMSELVES, not on the student paper>",
  "lang_style_rubric_feedback": "<2–3 sentences of feedback on the Language & Style rubric criteria THEMSELVES, not on the student paper>"
}}""".strip()


def _build_holistic_prompt_improved(rubric_text: str) -> str:
    """
    System prompt for the improved rubric holistic grading call.

    The model reads the entire paper and assigns:
      - A grade per section (Introduction, Methods, Results, Discussion)
      - A Language & Style grade
      - Rubric feedback per section + Language & Style
        (about the rubric as a tool, not about the student's paper)
    The final grade is computed in Python from these component grades.
    """
    return f"""{LLM_ROLE}
You will read the entire paper and assign a grade to each component separately.

## Grading instructions
Use the rubric below to grade each component. Five performance bands are provided for each
component, each with a corresponding grade range:
  - Insufficient : < 5.5
  - Developing   : 5.5 – 6.4
  - Adequate     : 6.5 – 7.4
  - Good         : 7.5 – 8.4
  - Excellent    : > 8.5

Assign a single numeric grade within the appropriate band for each component.
Grades must be between 1.0 and 10.0 in steps of 0.5. The grade 5.5 is NOT allowed —
use 5.0 (insufficient) or 6.0 (lowest passing) instead.

The final grade will be computed automatically from your component grades using the
weighted formula specified in the rubric. You do NOT need to compute it yourself.

## Rubric feedback instructions
After grading, provide brief constructive feedback on the rubric CRITERIA THEMSELVES
for each component. Imagine you are a rubric designer reviewing your own tool. Note any
criteria that are ambiguous, band descriptors that are hard to distinguish from each other,
or aspects that would benefit from more concrete examples or clearer thresholds.
Do NOT describe or evaluate the student's work in this field.

## Full grading rubric
{rubric_text}

## Output format
Respond with a JSON object containing EXACTLY these keys — no more, no less:
{{
  "intro_grade": <number>,
  "methods_grade": <number>,
  "results_grade": <number>,
  "discussion_grade": <number>,
  "lang_style_grade": <number>,
  "intro_rubric_feedback": "<2–3 sentences of feedback on the Introduction rubric criteria THEMSELVES, not on the student paper>",
  "methods_rubric_feedback": "<2–3 sentences of feedback on the Methods rubric criteria THEMSELVES, not on the student paper>",
  "results_rubric_feedback": "<2–3 sentences of feedback on the Results rubric criteria THEMSELVES, not on the student paper>",
  "discussion_rubric_feedback": "<2–3 sentences of feedback on the Discussion rubric criteria THEMSELVES, not on the student paper>",
  "lang_style_rubric_feedback": "<2–3 sentences of feedback on the Language & Style rubric criteria THEMSELVES, not on the student paper>"
}}""".strip()


# API call wrapper part

def _call_openai(
    system_prompt: str,
    user_content:  str,
    model:         str,
    temperature:   float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


# CSV-writing part

def append_result_row(csv_path: str, row: dict) -> None:
    """Append one row to the CSV, writing the header first if the file is new."""
    file_exists = os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# The grading function itself

def grade_paper_holistic(
    student_id:     str,
    sections:       dict[str, str],
    rubric_text:    str,
    rubric_version: str,
    model:          str,
    temperature:    float,
    run_id:         str,
) -> dict:
    """
    Grade one paper in a single API call (holistic pipeline).
    Returns a flat dict ready to be written as one CSV row.

    Column names match sequential_grading.py where applicable so that
    both scripts feed cleanly into the same JASP dataset.
    """
    print(f"\n{'─' * 64}")
    print(f"  run_id   : {run_id}")
    print(f"  model    : {model}  |  rubric: {rubric_version}  |  temp: {temperature}")
    print(f"{'─' * 64}")

    full_paper_text = build_full_paper_text(sections)
    token_count     = check_token_count(full_paper_text, label=f"{student_id}.full_paper")
    print(f"  Full paper: {token_count} tokens.")

    # Build prompt
    if rubric_version == "original":
        system_prompt = _build_holistic_prompt_original(rubric_text)
    else:
        system_prompt = _build_holistic_prompt_improved(rubric_text)

    user_content = (
        "Please grade the following student paper according to your instructions.\n\n"
        + full_paper_text
    )

    # Prompt preview
    print(f"\n  ┌─ PROMPT SENT (rubric: {rubric_version}) {'─' * 30}")
    print(f"  │  [system — first 600 chars]")
    for line in system_prompt[:600].splitlines():
        print(f"  │  {line}")
    if len(system_prompt) > 600:
        print(f"  │  ... [truncated — full prompt is {len(system_prompt)} chars]")
    print(f"  │  [user — paper starts with]")
    print(f"  │  {full_paper_text[:200].replace(chr(10), ' ')}")
    print(f"  └{'─' * 60}")

    # API call
    raw_reply = _call_openai(system_prompt, user_content, model, temperature)

    # Parse reply
    if rubric_version == "original":
        parsed = parse_holistic_reply_original(raw_reply)

        final_grade         = parsed["final_grade"]
        final_justification = parsed["justification"]

        intro_grade      = None
        methods_grade    = None
        results_grade    = None
        discussion_grade = None
        lang_style_grade = None

        print(f"  ✓ Holistic grading done (original rubric).")
        print(f"    final_grade   : {final_grade}")
        print(f"    justification : {str(final_justification or '')[:120]}")
        if final_grade is None:
            print(f"  ⚠  Could not parse a valid final grade. Raw reply:")
            print(f"     {raw_reply[:300]}")

    else:  # improved rubric
        parsed = parse_holistic_reply_improved(raw_reply)

        intro_grade      = parsed["intro_grade"]
        methods_grade    = parsed["methods_grade"]
        results_grade    = parsed["results_grade"]
        discussion_grade = parsed["discussion_grade"]
        lang_style_grade = parsed["lang_style_grade"]

        component_grades = {
            "introduction":   intro_grade,
            "methods":        methods_grade,
            "results":        results_grade,
            "discussion":     discussion_grade,
            "language_style": lang_style_grade,
        }
        final_grade         = compute_improved_final_grade(component_grades)
        final_justification = None

        print(f"  ✓ Holistic grading done (improved rubric).")
        print(f"    intro_grade      : {intro_grade}")
        print(f"    methods_grade    : {methods_grade}")
        print(f"    results_grade    : {results_grade}")
        print(f"    discussion_grade : {discussion_grade}")
        print(f"    lang_style_grade : {lang_style_grade}")
        print(f"    final_grade      : {final_grade}  (weighted)")
        if final_grade is None:
            print("  ⚠  Could not compute final grade — one or more component grades are missing.")
            print(f"     Raw reply: {raw_reply[:300]}")

    # Make CSV rows
    row: dict = {
        "run_id":                     run_id,
        "student_id":                 student_id,
        "grading_mode":               "holistic",
        "rubric_version":             rubric_version,
        "model":                      model,
        "temperature":                temperature,
        "timestamp":                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "intro_grade":                intro_grade,
        "intro_content_summary":      None,
        "intro_style_notes":          None,
        "intro_rubric_feedback":      parsed.get("intro_rubric_feedback"),
        "methods_grade":              methods_grade,
        "methods_content_summary":    None,
        "methods_style_notes":        None,
        "methods_rubric_feedback":    parsed.get("methods_rubric_feedback"),
        "results_grade":              results_grade,
        "results_content_summary":    None,
        "results_style_notes":        None,
        "results_rubric_feedback":    parsed.get("results_rubric_feedback"),
        "discussion_grade":           discussion_grade,
        "discussion_content_summary": None,
        "discussion_style_notes":     None,
        "discussion_rubric_feedback": parsed.get("discussion_rubric_feedback"),
        "lang_style_grade":           lang_style_grade,
        "lang_style_rubric_feedback": parsed.get("lang_style_rubric_feedback"),
        "final_grade":                final_grade,
        "final_grade_justification":  final_justification,
    }

    return row


# Checks whether papers have already been graded and skips them if so; prints finalization message

if __name__ == "__main__":

    RUBRICS   = load_rubrics()
    PIPELINES = load_pipelines(PIPELINES_FILE)

    # Resume support
    completed: set[str] = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            completed = {row["run_id"] for row in reader if "run_id" in row}
    print(f"Skipping {len(completed)} already completed run(s).")

    papers = list_papers(PAPERS_FOLDER)
    print(f"Found {len(papers)} paper(s): {[sid for sid, _ in papers]}\n")

    total_runs = len(papers) * len(PIPELINES) * N_RUNS
    done = 0

    for student_id, doc_path in papers:

        try:
            sections  = load_document(doc_path)
            full_text = build_full_paper_text(sections)
            print(f"\nPaper '{student_id}': sections found → {list(sections.keys())}")
            check_token_count(full_text, label=f"{student_id}.full_paper")
        except Exception as e:
            print(f"  ✗ Cannot load '{student_id}': {e} — skipping.")
            continue

        for pipeline in PIPELINES:
            rubric_version = pipeline["rubric_version"]
            model          = pipeline["model"]
            temperature    = pipeline.get("temperature", 1.0)

            for run_idx in range(1, N_RUNS + 1):
                run_id = f"{student_id}.{pipeline['pipeline_id']}.run{run_idx}"

                if run_id in completed:
                    print(f"  → Skipping {run_id} (already done).")
                    done += 1
                    continue

                try:
                    row = grade_paper_holistic(
                        student_id     = student_id,
                        sections       = sections,
                        rubric_text    = RUBRICS[rubric_version],
                        rubric_version = rubric_version,
                        model          = model,
                        temperature    = temperature,
                        run_id         = run_id,
                    )
                    append_result_row(RESULTS_CSV, row)
                    done += 1
                    print(f"\n  ✓ Saved {run_id}  [{done}/{total_runs}]")

                except Exception as e:
                    print(f"\n  ✗ Error on {run_id}: {e}")

    print(f"\n{'═' * 64}")
    print(f"  Holistic grading complete.")
    print(f"  Results saved to: {RESULTS_CSV}")
    print(f"{'═' * 64}")
