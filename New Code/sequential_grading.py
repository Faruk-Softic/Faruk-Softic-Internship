"""
This script grades papers by grading each section in a separate API-call, using summaries made in each preceding
part as context for grading the current section. There is no context needed for the introduction.
Lastly, grading the discussion uses the summary from the introduction, instead of the results section.
"""

import os
import csv
import json
import glob
import re
import pdfplumber
import tiktoken
from openai import OpenAI
from datetime import datetime
from docx import Document
from typing import Optional

# If needed, configure paths here

API_KEY_FILE     = r"C:\Users\fsoft\Desktop\api_key.txt"
PAPERS_FOLDER    = r"C:\Users\fsoft\Desktop\Za Internship - code\Papers"
RUBRIC_ORIGINAL  = r"C:\Users\fsoft\Desktop\Za Internship - code\Rubric_original.docx"
RUBRIC_IMPROVED  = r"C:\Users\fsoft\Desktop\Za Internship - code\Rubric_improved.docx"
PIPELINES_FILE   = r"C:\Users\fsoft\Desktop\Za Internship - code\pipelines.json"
RESULTS_CSV      = r"C:\Users\fsoft\Desktop\Za Internship - code\results.csv"
# If needed, change number of runs here

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


# API-client and key

def load_api_key(path: str) -> str:
    """Read the OpenAI API key from a plain-text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

client = OpenAI(api_key=load_api_key(API_KEY_FILE))


# Text extration for both docx and pdf file types

def read_pdf(path: str) -> str:
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


def read_docx(path: str) -> str:
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


def load_document(path: str) -> dict[str, str]:
    """Detect file type, extract text, and split into sections."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".pdf":
        full_text = read_pdf(path)
    elif ext == ".docx":
        full_text = read_docx(path)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .pdf or .docx.")
    return split_into_sections(full_text)


def split_into_sections(text: str) -> dict[str, str]:
    """
    This part splits the document into four sections as defined by subheadings
    to be used for grading, accounting for some variations in the subheading itself.
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


# Checks token count and warns upon approaching 100.000

ENCODING   = tiktoken.get_encoding("o200k_base")
MAX_TOKENS = 100_000

def check_token_count(text: str, label: str = "") -> int:
    n = len(ENCODING.encode(text))
    if n > MAX_TOKENS:
        raise ValueError(
            f"[{label}] Text is {n} tokens, exceeding the limit of {MAX_TOKENS}."
        )
    if n > MAX_TOKENS * 0.85:
        print(f"  ⚠  Warning: [{label}] is {n} tokens (>85 % of limit).")
    return n


# This part loads both rubrics files and keeps them in memory

def load_rubrics() -> dict[str, str]:
    """
    Load both rubric files once and keep them in memory.
    Reused for every paper and run — files are never read more than twice.
    """
    print("Loading rubrics into memory cache...")
    rubrics = {
        "original": read_docx(RUBRIC_ORIGINAL),
        "improved": read_docx(RUBRIC_IMPROVED),
    }
    for name, text in rubrics.items():
        n = check_token_count(text, label=f"rubric_{name}")
        print(f"  ✓ Rubric '{name}' loaded ({n} tokens).")
    return rubrics


# ─This part loads pipelines and individual papers

def load_pipelines(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_papers(folder: str) -> list[tuple[str, str]]:
    """Return (student_id, path) pairs for all .pdf and .docx files in folder."""
    paths = (
        glob.glob(os.path.join(folder, "*.pdf")) +
        glob.glob(os.path.join(folder, "*.docx"))
    )
    papers = []
    for path in paths:
        filename = os.path.basename(path)
        if filename.startswith("~$"):
            continue
        student_id = os.path.splitext(filename)[0]
        papers.append((student_id, path))
    return sorted(papers, key=lambda x: x[0])


# This part parses grades and rounds them (REMEMBER TO REVISE THIS TO FIX THE ROUNDING)

def parse_grade(value) -> Optional[float]:
    """
    Validate and normalise a grade value to the Dutch grading scale:
      - Range  : 1.0 – 10.0
      - Step   : 0.5
      - 5.5 is not a valid Dutch grade → rounded down to 5.0
    """
    if value is None:
        return None
    try:
        g = float(value)
    except (TypeError, ValueError):
        return None
    g = round(g * 2) / 2.0      # snap to nearest 0.5
    if g < 1.0 or g > 10.0:
        return None
    if g == 5.5:
        g = 5.0                  # 5.5 not valid on Dutch scale → round down
    return g


def parse_section_reply(reply: str) -> dict:
    """
    Parse the JSON reply for a single section.
    Expected keys: grade, content_summary, style_notes, rubric_feedback
    """
    try:
        data = json.loads(reply)
    except Exception:
        return {"grade": None, "content_summary": None,
                "style_notes": None, "rubric_feedback": None,
                "raw_reply": reply}
    return {
        "grade":           parse_grade(data.get("grade")),
        "content_summary": data.get("content_summary"),
        "style_notes":     data.get("style_notes"),
        "rubric_feedback": data.get("rubric_feedback"),
        "raw_reply":       reply,
    }


def parse_language_style_reply(reply: str) -> dict:
    """
    Parse the JSON reply for the Language & Style step.
    Expected keys: grade, rubric_feedback
    """
    try:
        data = json.loads(reply)
    except Exception:
        return {"grade": None, "rubric_feedback": None, "raw_reply": reply}
    return {
        "grade":           parse_grade(data.get("grade")),
        "rubric_feedback": data.get("rubric_feedback"),
        "raw_reply":       reply,
    }


def parse_holistic_final_reply(reply: str) -> dict:
    """
    Parse the JSON reply for the holistic final grade (original rubric only).
    Expected keys: final_grade, justification
    """
    try:
        data = json.loads(reply)
    except Exception:
        return {"final_grade": None, "justification": None, "raw_reply": reply}
    return {
        "final_grade":    parse_grade(data.get("final_grade")),
        "justification":  data.get("justification"),
        "raw_reply":      reply,
    }


# This part calculates final grades as defined by the weighted formula in the improved rubrics

def compute_improved_final_grade(section_grades: dict[str, Optional[float]]) -> Optional[float]:
    """
    Compute the weighted final grade for the improved rubric.
    Returns None if any required grade is missing.

    Caps:
      - Any component < 5.5  → final grade cannot exceed 6.0
      - Two or more < 5.5    → final grade cannot exceed 5.0
      - 5.5 is not valid     → rounded down to 5.0
    """
    required = list(IMPROVED_WEIGHTS.keys())
    for key in required:
        if section_grades.get(key) is None:
            return None

    weighted_sum = sum(
        section_grades[key] * IMPROVED_WEIGHTS[key] for key in required
    )
    final = round(weighted_sum * 2) / 2.0

    below_threshold = [k for k in required if section_grades[k] < 5.5]
    if len(below_threshold) >= 2:
        final = min(final, 5.0)
    elif len(below_threshold) == 1:
        final = min(final, 6.0)

    if final == 5.5:
        final = 5.0

    return final


# Prompt-building part

def build_section_prompt(
    section_name:       str,
    rubric_text:        str,
    previous_summaries: dict[str, dict],
    rubric_version:     str,
) -> str:
    """
    Build the system prompt for grading one section.

    Context passed forward:
      introduction → (no prior context)
      methods      → introduction content_summary
      results      → methods content_summary
      discussion   → introduction content_summary + style_notes,
                     results content_summary
    """

    # This part creates the summaries to be used as context in subsequent section gradings
    summary_block = ""
    if section_name == "methods" and "introduction" in previous_summaries:
        s = previous_summaries["introduction"]
        summary_block = (
            "## Context from previously graded sections\n\n"
            f"**Introduction — content summary:**\n{s.get('content_summary', '')}\n"
        )
    elif section_name == "results" and "methods" in previous_summaries:
        s = previous_summaries["methods"]
        summary_block = (
            "## Context from previously graded sections\n\n"
            f"**Methods — content summary:**\n{s.get('content_summary', '')}\n"
        )
    elif section_name == "discussion":
        parts = ["## Context from previously graded sections\n"]
        if "introduction" in previous_summaries:
            s = previous_summaries["introduction"]
            parts.append(
                f"**Introduction — content summary:**\n{s.get('content_summary', '')}\n\n"
                f"**Introduction — style notes:**\n{s.get('style_notes', '')}\n"
            )
        if "results" in previous_summaries:
            s = previous_summaries["results"]
            parts.append(
                f"**Results — content summary:**\n{s.get('content_summary', '')}\n"
            )
        summary_block = "\n".join(parts)

    # This part instructs LLM to use the appropriate context summary
    instructions = {
        "introduction": (
            "Grade the Introduction section using the rubric provided. "
            "Pay attention to whether the research question is clearly stated and justified, "
            "whether the theoretical reasoning is sound, and whether the hypotheses are "
            "testable and logically derived from the introduction."
        ),
        "methods": (
            "Grade the Methods section using the rubric provided. "
            "Keep in mind the Introduction content summary above: check whether the "
            "methodology is consistent with and appropriate for the stated research question "
            "and hypotheses."
        ),
        "results": (
            "Grade the Results section using the rubric provided. "
            "Keep in mind the Methods content summary above: check whether the results "
            "are reported in a way that matches the described methodology."
        ),
        "discussion": (
            "Grade the Discussion section using the rubric provided. "
            "This section must be evaluated in light of the Introduction: check whether "
            "the conclusions are consistent with the original reasoning and hypotheses, "
            "and whether the discussion adequately addresses the research question. "
            "Also refer to the Introduction style notes when assessing writing quality."
        ),
    }
    section_instruction = instructions.get(section_name, "Grade this section using the rubric.")

    # This part instructs LLM to provide feedback on the rubrics themselves, for improvement purposes
    if rubric_version == "improved":
        feedback_instruction = (
            "Additionally, provide brief constructive feedback on the rubric CRITERIA "
            "THEMSELVES — not on the student's paper. Imagine you are a rubric designer "
            "reviewing your own tool. Note any criteria that are ambiguous, band descriptors "
            "that are hard to distinguish from each other, or aspects that would benefit from "
            "more concrete examples or clearer thresholds. "
            "Do NOT describe or evaluate the student's work in this field."
        )
    else:
        feedback_instruction = (
            "Additionally, provide brief feedback on the rubric CRITERIA THEMSELVES — "
            "not on the student's paper. Imagine you are a rubric designer reviewing your "
            "own tool. Note any criteria that are unclear, overly subjective, or that would "
            "benefit from more concrete anchors or clearer band descriptors. "
            "Do NOT describe or evaluate the student's work in this field."
        )

    prompt = f"""{LLM_ROLE}
You will grade one section of the paper at a time. You have access to the full grading rubric below.

{summary_block}
## Rubric
{rubric_text}

## Your task for this section: {section_name.upper()}
{section_instruction}

{feedback_instruction}

## Style notes instruction
As part of your summary, you must also write dedicated style notes for this section.
Style notes should cover: clarity and precision of language, grammar and spelling,
APA formatting compliance, paragraph structure, and any redundancy or inconsistency
in writing. These notes will be used later to assign an overall Language & Style grade
for the full paper.

## Output format
Respond with a JSON object containing exactly these keys:
{{
  "grade": <number between 1.0 and 10.0, multiples of 0.5, not 5.5>,
  "content_summary": "<3–5 sentence summary of the scientific content and quality of this section>",
  "style_notes": "<2–4 sentence summary of language and style observations for this section>",
  "rubric_feedback": "<2–4 sentences of constructive feedback on the rubric criteria THEMSELVES, not on the student paper>"
}}
"""
    return prompt.strip()


def build_language_style_prompt(
    rubric_text:     str,
    all_style_notes: dict[str, str],
    rubric_version:  str,
) -> str:
    """
    Build the system prompt for the Language & Style grading step.
    The LLM receives style notes from all four sections.
    """
    notes_block = "\n\n".join(
        f"**{sec.capitalize()} — style notes:**\n{note}"
        for sec, note in all_style_notes.items()
        if note
    )

    if rubric_version == "improved":
        feedback_instruction = (
            "Additionally, provide brief constructive feedback on the Language & Style "
            "rubric CRITERIA THEMSELVES — not on the student's paper. Imagine you are a "
            "rubric designer reviewing your own tool. Note any criteria that are ambiguous, "
            "thresholds that are hard to apply, or aspects that would benefit from more "
            "concrete examples or clearer descriptors. "
            "Do NOT describe or evaluate the student's work in this field."
        )
    else:
        feedback_instruction = (
            "Additionally, provide brief feedback on the Language & Style rubric CRITERIA "
            "THEMSELVES — not on the student's paper. Imagine you are a rubric designer "
            "reviewing your own tool. Note any criteria that are unclear, overly subjective, "
            "or that would benefit from more concrete anchors or clearer band descriptors. "
            "Do NOT describe or evaluate the student's work in this field."
        )

    prompt = f"""{LLM_ROLE}
You have already graded all four sections of the paper (Introduction, Methods, Results, Discussion).
Below are your style notes from each section, and the full grading rubric.

## Style notes from all sections
{notes_block}

## Rubric
{rubric_text}

## Your task
Using the style notes above and the Language & Style criteria in the rubric, assign an overall
Language & Style grade for the entire paper. Base your grade on patterns observed across all
sections, not just one.

{feedback_instruction}

## Output format
Respond with a JSON object containing exactly these keys:
{{
  "grade": <number between 1.0 and 10.0, multiples of 0.5, not 5.5>,
  "rubric_feedback": "<2–4 sentences of constructive feedback on the Language & Style rubric criteria THEMSELVES, not on the student paper>"
}}
"""
    return prompt.strip()


def build_holistic_final_prompt(
    rubric_text:     str,
    section_results: dict[str, dict],
) -> str:
    """
    Build the prompt for the holistic final grade (original rubric only).
    The LLM receives all section grades and summaries.
    """
    summaries_block = ""
    for sec in SECTION_ORDER:
        if sec in section_results:
            r = section_results[sec]
            summaries_block += (
                f"**{sec.capitalize()}** (section grade: {r.get('grade')}):\n"
                f"{r.get('content_summary', '')}\n\n"
            )

    prompt = f"""{LLM_ROLE}
You have graded all sections individually. Now you must assign a single holistic final grade
for the entire paper, as required by the rubric.

The rubric explicitly states that the paper should be graded as a whole, not as a weighted
average of section scores. Use the section grades and summaries below only as reference points
to inform your holistic judgment.

## Section grades and summaries
{summaries_block}

## Rubric
{rubric_text}

## Your task
Assign a single holistic final grade for the paper. Justify your grade in 3–5 sentences,
explaining how the overall quality of the paper maps onto the rubric's performance bands.

## Output format
Respond with a JSON object containing exactly these keys:
{{
  "final_grade": <number between 1.0 and 10.0, multiples of 0.5, not 5.5>,
  "justification": "<3–5 sentence justification of the holistic final grade>"
}}
"""
    return prompt.strip()


# API call wrapper

def call_openai(
    system_prompt: str,
    user_content:  str,
    model:         str,
    temperature:   float,
) -> str:
    """Send one chat completion request and return the raw reply string."""
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


# CSV writer

def append_result_row(csv_path: str, row: dict) -> None:
    """Append one result row to the CSV, writing the header if the file is new."""
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# This part defines the core grading function

def grade_paper_sequential(
    student_id:     str,
    sections:       dict[str, str],
    rubric_text:    str,
    rubric_version: str,
    model:          str,
    temperature:    float,
    run_id:         str,
) -> dict:
    """
    Grade one paper sequentially through all sections, then Language & Style,
    then compute or request the final grade.
    Returns a flat dict ready to be written as a CSV row.
    """
    print(f"\n{'─' * 60}")
    print(f"  Grading: {run_id}")
    print(f"  Model: {model} | Rubric: {rubric_version} | Temp: {temperature}")
    print(f"{'─' * 60}")

    previous_summaries: dict[str, dict] = {}
    section_results:    dict[str, dict] = {}

    # Grade each section
    for sec_name in SECTION_ORDER:
        if sec_name not in sections:
            print(f"  ⚠  Section '{sec_name}' not found in paper — skipping.")
            section_results[sec_name] = {
                "grade": None, "content_summary": None,
                "style_notes": None, "rubric_feedback": None,
                "raw_reply": None,
            }
            continue

        sec_text      = sections[sec_name]
        system_prompt = build_section_prompt(
            section_name       = sec_name,
            rubric_text        = rubric_text,
            previous_summaries = previous_summaries,
            rubric_version     = rubric_version,
        )
        user_content = (
            f"Please grade the following {sec_name.upper()} section:\n\n{sec_text}"
        )

        print(f"\n  ┌─ PROMPT SENT FOR {sec_name.upper()} {'─' * 30}")
        print(f"  │ SYSTEM:\n{system_prompt[:800]}{'...[truncated]' if len(system_prompt) > 800 else ''}")
        print(f"  │ USER (first 300 chars):\n{user_content[:300]}...")
        print(f"  └{'─' * 50}")

        raw_reply = call_openai(system_prompt, user_content, model, temperature)
        result    = parse_section_reply(raw_reply)
        section_results[sec_name] = result

        print(f"  ✓ {sec_name.upper()} graded.")
        print(f"    Grade          : {result['grade']}")
        print(f"    Content summary: {str(result['content_summary'])[:120]}...")
        print(f"    Style notes    : {str(result['style_notes'])[:120]}...")
        print(f"    Rubric feedback: {str(result['rubric_feedback'])[:120]}...")
        if result["grade"] is None:
            print(f"  ⚠  Grade could not be parsed for {sec_name}. Raw reply:\n{raw_reply}")

        previous_summaries[sec_name] = {
            "content_summary": result["content_summary"],
            "style_notes":     result["style_notes"],
        }

    # Language and Style
    all_style_notes = {
        sec: section_results[sec]["style_notes"]
        for sec in SECTION_ORDER
        if section_results[sec]["style_notes"]
    }
    ls_prompt = build_language_style_prompt(rubric_text, all_style_notes, rubric_version)
    ls_user   = (
        "Using the style notes above and the rubric, assign an overall "
        "Language & Style grade for the entire paper."
    )

    print(f"\n  ┌─ PROMPT SENT FOR LANGUAGE & STYLE {'─' * 20}")
    print(f"  │ SYSTEM:\n{ls_prompt[:600]}{'...[truncated]' if len(ls_prompt) > 600 else ''}")
    print(f"  └{'─' * 50}")

    ls_raw    = call_openai(ls_prompt, ls_user, model, temperature)
    ls_result = parse_language_style_reply(ls_raw)

    print(f"  ✓ Language & Style graded.")
    print(f"    Grade          : {ls_result['grade']}")
    print(f"    Rubric feedback: {str(ls_result['rubric_feedback'])[:120]}...")

    # Final grade
    final_grade         = None
    final_justification = None

    if rubric_version == "improved":
        all_grades = {sec: section_results[sec]["grade"] for sec in SECTION_ORDER}
        all_grades["language_style"] = ls_result["grade"]
        final_grade = compute_improved_final_grade(all_grades)
        print(f"\n  ✓ Improved rubric weighted final grade: {final_grade}")

    else:
        hf_prompt = build_holistic_final_prompt(rubric_text, section_results)
        hf_user   = "Please assign a holistic final grade for the paper."

        print(f"\n  ┌─ PROMPT SENT FOR HOLISTIC FINAL GRADE {'─' * 15}")
        print(f"  │ SYSTEM:\n{hf_prompt[:600]}{'...[truncated]' if len(hf_prompt) > 600 else ''}")
        print(f"  └{'─' * 50}")

        hf_raw    = call_openai(hf_prompt, hf_user, model, temperature)
        hf_result = parse_holistic_final_reply(hf_raw)
        final_grade         = hf_result["final_grade"]
        final_justification = hf_result["justification"]

        print(f"  ✓ Original rubric holistic final grade: {final_grade}")
        print(f"    Justification: {str(final_justification)[:120]}...")

    # Make CSV rows
    row = {
        "run_id":                     run_id,
        "student_id":                 student_id,
        "grading_mode":               "sequential",
        "rubric_version":             rubric_version,
        "model":                      model,
        "temperature":                temperature,
        "timestamp":                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "intro_grade":                section_results["introduction"]["grade"],
        "intro_content_summary":      section_results["introduction"]["content_summary"],
        "intro_style_notes":          section_results["introduction"]["style_notes"],
        "intro_rubric_feedback":      section_results["introduction"]["rubric_feedback"],
        "methods_grade":              section_results["methods"]["grade"],
        "methods_content_summary":    section_results["methods"]["content_summary"],
        "methods_style_notes":        section_results["methods"]["style_notes"],
        "methods_rubric_feedback":    section_results["methods"]["rubric_feedback"],
        "results_grade":              section_results["results"]["grade"],
        "results_content_summary":    section_results["results"]["content_summary"],
        "results_style_notes":        section_results["results"]["style_notes"],
        "results_rubric_feedback":    section_results["results"]["rubric_feedback"],
        "discussion_grade":           section_results["discussion"]["grade"],
        "discussion_content_summary": section_results["discussion"]["content_summary"],
        "discussion_style_notes":     section_results["discussion"]["style_notes"],
        "discussion_rubric_feedback": section_results["discussion"]["rubric_feedback"],
        "lang_style_grade":           ls_result["grade"],
        "lang_style_rubric_feedback": ls_result["rubric_feedback"],
        "final_grade":                final_grade,
        "final_grade_justification":  final_justification,
    }

    return row


# This part checks if the current paper has already been graded and skips it if so, and prints a finalization message.

if __name__ == "__main__":

    RUBRICS   = load_rubrics()
    PIPELINES = load_pipelines(PIPELINES_FILE)

    PIPELINES = [p for p in PIPELINES if p.get("grading_mode") == "sequential"]
    print(f"Loaded {len(PIPELINES)} sequential pipeline(s).")

    completed: set[str] = set()
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            completed = {row["run_id"] for row in reader}
    print(f"Skipping {len(completed)} already completed run(s).")

    papers = list_papers(PAPERS_FOLDER)
    print(f"Found {len(papers)} paper(s): {[s for s, _ in papers]}")

    total = len(papers) * len(PIPELINES) * N_RUNS
    done  = 0

    for student_id, document_path in papers:

        try:
            sections = load_document(document_path)
            print(f"\n  Paper '{student_id}': found sections {list(sections.keys())}")
            for sec_name, sec_text in sections.items():
                check_token_count(sec_text, label=f"{student_id}.{sec_name}")
        except Exception as e:
            print(f"  ✗ Could not load '{student_id}': {e} — skipping.")
            continue

        for pipeline in PIPELINES:
            rubric_text    = RUBRICS[pipeline["rubric_version"]]
            rubric_version = pipeline["rubric_version"]
            model          = pipeline["model"]
            temperature    = pipeline.get("temperature", 1.0)

            for run_idx in range(1, N_RUNS + 1):
                run_id = (
                    f"{student_id}"
                    f".{pipeline['pipeline_id']}"
                    f".run{run_idx}"
                )

                if run_id in completed:
                    print(f"  → Skipping {run_id} (already done).")
                    done += 1
                    continue

                try:
                    row = grade_paper_sequential(
                        student_id     = student_id,
                        sections       = sections,
                        rubric_text    = rubric_text,
                        rubric_version = rubric_version,
                        model          = model,
                        temperature    = temperature,
                        run_id         = run_id,
                    )
                    append_result_row(RESULTS_CSV, row)
                    done += 1
                    print(f"\n  ✓ Saved {run_id}  [{done}/{total}]")

                except Exception as e:
                    print(f"\n  ✗ Error on {run_id}: {e}")

    print(f"\n{'═' * 60}")
    print(f"  Sequential grading complete.")
    print(f"  Results saved to: {RESULTS_CSV}")
    print(f"{'═' * 60}")
