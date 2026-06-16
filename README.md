# Faruk-Softic-Internship

This repository contains the code and data for a study investigating whether LLM-based grading of student psychology papers is consistent, and whether rubric design affects grading outcomes. The code is provided, but course materials and student papers are not shared.

---

## Repository Structure

```
/
├── New Code/
│   ├── synchronous.py        # Grades papers one at a time via the OpenAI Responses API
│   └── batch.py              # Grades papers in bulk via the OpenAI Batch API
│
├── r_analysis.R              # Combines results.csv with teacher grades; produces scatterplot
├── pipelines.json            # Defines grading pipelines (model, rubric version, temperature)
├── allinfo_pseudo.csv        # Pseudonymised original course grades
├── results.csv               # LLM grading output produced by the grading scripts
├── combined.csv              # Merged file produced by r_analysis.R
├── Rubric_original.docx      # Original grading rubric
└── Rubric_improved.docx      # Improved grading rubric
```

The following files and folders are excluded from the repository via `.gitignore`:

| Excluded | Reason |
|---|---|
| `Papers/` | Contains student papers |
| `Writing_guide.pdf`, `Sample_results.pdf` | Course materials |
| `Calibration_summary.docx` | Course materials |
| `Older versions/` | Previous iterations of the code |
| `combined.jasp` | JASP analysis file |
| `.env` | Contains secrets and local paths |

---

## Setup

### 1. Install dependencies

```bash
pip install openai pdfplumber python-docx tiktoken python-dotenv
```

### 2. Configure environment variables

Create a `.env` file in the root directory with the following variables:

```env
OPENAI_API_KEY=your_openai_api_key

PAPERS_FOLDER=/path/to/Papers
RUBRIC_ORIGINAL_PATH=/path/to/Rubric_original.docx
RUBRIC_IMPROVED_PATH=/path/to/Rubric_improved.docx
WRITING_GUIDE=/path/to/Writing_guide.pdf
SAMPLE_RESULTS=/path/to/Sample_results.pdf
CALIBRATION_SUMMARY=/path/to/Calibration_summary.docx
PIPELINES_FILE=/path/to/pipelines.json
RESULTS_CSV=/path/to/results.csv
```

### 3. Configure pipelines

`pipelines.json` defines which model, rubric version, and temperature to use. Each entry should follow this structure:

```json
[
  {
    "pipeline_id": "gpt4o_original",
    "grading_mode": "holistic",
    "rubric_version": "original",
    "model": "gpt-4o",
    "temperature": 1.0
  }
]
```

---

## Usage

### Synchronous (small batches / testing)

Grades papers one at a time. Progress is printed to the terminal in real time.

```bash
python "New Code/synchronous.py"
```

### Batch API (full runs)

Submits all papers as a single batch job, polls for completion, then parses and saves results. Recommended for large runs — costs ~50% less than the synchronous API.

```bash
python "New Code/batch.py"
```

Both scripts support **resume**: if `results.csv` already contains completed `run_id`s, those runs are skipped automatically.

---

## Output

Both scripts write to `results.csv` with the following columns:

| Column | Description |
|---|---|
| `run_id` | Unique identifier: `{student_id}.{pipeline_id}.run{n}` |
| `student_id` | Paper identifier |
| `grading_mode` | Always `holistic` |
| `rubric_version` | `original` or `improved` |
| `model` | OpenAI model used |
| `temperature` | Sampling temperature |
| `timestamp` | Date and time of the run |
| `intro_grade` … `lang_style_grade` | Section grades (1.0–10.0 in 0.5 steps) |
| `intro_feedback` … `lang_style_feedback` | LLM reasoning per section |
| `final_grade` | Weighted average of section grades |
| `missing_sections` | Comma-separated list of sections the LLM could not grade |

### Grade weights

| Section | Weight |
|---|---|
| Introduction | 30% |
| Methods | 15% |
| Results | 15% |
| Discussion | 30% |
| Language & Style | 10% |

---

## Analysis

`r_analysis.R` merges `results.csv` with `allinfo_pseudo.csv` (which contains the original teacher grades) into `combined.csv`, and produces a preliminary scatterplot comparing LLM grades to teacher grades.
