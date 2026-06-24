import re
from pathlib import Path

# Defining sections and their keys

SECTION_KEYS = [
    "intro",
    "methods",
    "results",
    "discussion",
    "lang_style",
]

SECTION_LABELS = {
    "intro":      "Introduction",
    "methods":    "Methods",
    "results":    "Results",
    "discussion": "Discussion",
    "lang_style": "Language & Style",
}

SECTION_WEIGHTS = {
    "intro":      0.30,
    "methods":    0.15,
    "results":    0.15,
    "discussion": 0.30,
    "lang_style": 0.10,
}

# Checklist keys

CHECKLIST = {
    "intro": [
        "intro_societal_relevance_present",
        "intro_theoretical_framework_present",
        "intro_prior_research_discussed",
        "intro_rq_and_hypothesis_stated",
        "intro_hypothesis_substantiated",
    ],
    "methods": [
        "methods_participants_described",
        "methods_materials_described",
        "methods_procedure_described",
        "methods_design_matches_rq",
    ],
    "results": [
        "results_assumptions_checked",
        "results_main_analysis_reported",
        "results_statistics_complete",
        "results_tables_or_figures_present",
    ],
    "discussion": [
        "discussion_main_conclusion_stated",
        "discussion_results_linked_to_theory",
        "discussion_comparison_to_prior_research",
        "discussion_limitations_discussed",
        "discussion_broader_implications_stated",
    ],
    "lang_style": [
        "lang_apa_referencing_correct",
        "lang_paragraph_structure_clear",
        "lang_scientific_language_used",
    ],
}

CHECKLIST_KEYS = [key for keys in CHECKLIST.values() for key in keys]

# Output structure definition

def get_output_schema(include_checklist: bool, include_feedback: bool) -> list[tuple[str, str]]:
    """
    Returns an ordered list of (key, type_hint) pairs that define the expected
    JSON output from the model. Used both to build the prompt and to parse the reply.
    """
    fields = []
    for sec in SECTION_KEYS:
        if include_checklist:
            for key in CHECKLIST[sec]:
                fields.append((key, "true | false"))
        if include_feedback:
            fields.append((f"{sec}_feedback", '"<string>"'))
        fields.append((f"{sec}_grade", "<number>"))
    return fields

# CSV fieldnames - makes sure CSV output is in line with the schema above

def get_csv_fieldnames(include_checklist: bool, include_feedback: bool) -> list[str]:

    base = [
        "run_id", "run_label", "student_id", "pipeline_id", "repetition",
        "model", "reasoning_effort", "timestamp", "input_tokens", "output_tokens",
    ]
    for sec in SECTION_KEYS:
        base.append(f"{sec}_grade")
    base += ["final_grade", "missing_sections"]
    if include_feedback:
        for sec in SECTION_KEYS:
            base.append(f"{sec}_feedback")
    if include_checklist:
        base.extend(CHECKLIST_KEYS)
    return base

# Run folder detection

def get_latest_run_folder(base: Path) -> Path:
    folders = [
        d for d in base.iterdir()
        if d.is_dir() and re.fullmatch(r"run-\d+", d.name)
    ]
    if not folders:
        raise FileNotFoundError(f"No run folders found in '{base}'.")
    return max(folders, key=lambda d: int(d.name.split("-")[1]))
