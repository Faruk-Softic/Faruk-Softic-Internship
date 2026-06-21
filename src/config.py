# Section keys
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

# Checklist keys for the "improved" pipeline
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
