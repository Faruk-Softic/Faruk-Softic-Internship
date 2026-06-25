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

# Checklist keys — full tiered structure mirroring the rubric exactly

CHECKLIST = {
    "intro": [
        # Tier 0 knock-outs
        "intro_t0_no_rq",
        "intro_t0_no_hypothesis",
        "intro_t0_incoherent_reasoning",
        "intro_t0_fewer_than_3_articles",
        # Tier 1 core
        "intro_t1_rq_stated",
        "intro_t1_hypothesis_stated",
        "intro_t1_societal_relevance",
        "intro_t1_all_4_articles",
        "intro_t1_reasoning_present",
        "intro_t1_key_concepts_defined",
        # Tier 1 additional
        "intro_t1a_hypothesis_follows_logically",
        "intro_t1a_articles_connected",
        "intro_t1a_scientific_relevance_mentioned",
        # Tier 2 core
        "intro_t2_rq_clear_and_relevant",
        "intro_t2_hypothesis_substantiated",
        "intro_t2_articles_connected_to_rq",
        "intro_t2_scientific_relevance_argued",
        "intro_t2_theoretical_framework_present",
        # Tier 2 additional
        "intro_t2a_smooth_transitions",
        "intro_t2a_study_positioned_relative_to_gaps",
        "intro_t2a_theory_used_for_hypothesis",
        "intro_t2a_societal_relevance_well_aligned",
        # Tier 3 core
        "intro_t3_reasoning_coherent_and_convincing",
        "intro_t3_prior_research_as_building_blocks",
        "intro_t3_hypothesis_from_theory_and_research",
        "intro_t3_design_choice_justified",
        # Tier 3 additional
        "intro_t3a_articles_ordered_optimally",
        "intro_t3a_design_improves_on_prior",
        "intro_t3a_limitations_of_prior_acknowledged",
        "intro_t3a_peel_structure",
        # Tier 4 core
        "intro_t4_reasoning_very_coherent_and_tight",
        "intro_t4_reads_as_seamless_argument",
        "intro_t4_rq_compellingly_motivated",
        # Tier 4 additional
        "intro_t4a_framework_original_or_nuanced",
        "intro_t4a_counterarguments_addressed",
        "intro_t4a_mature_grasp_of_scientific_reasoning",
    ],
    "methods": [
        # Tier 0
        "methods_t0_participants_absent",
        "methods_t0_materials_absent",
        "methods_t0_procedure_absent",
        "methods_t0_incoherent_design",
        # Tier 1 core
        "methods_t1_participants_sample_size_and_criteria",
        "methods_t1_materials_iv_described",
        "methods_t1_materials_questionnaire_with_reference",
        "methods_t1_procedure_chronological",
        "methods_t1_design_coherent_with_rq",
        # Tier 1 additional
        "methods_t1a_allocation_described",
        "methods_t1a_questionnaire_items_and_range",
        "methods_t1a_no_procedural_language_in_materials",
        # Tier 2 core
        "methods_t2_participants_age_sd_gender_compensation",
        "methods_t2_participants_criteria_substantiated",
        "methods_t2_materials_all_questionnaires_complete",
        "methods_t2_materials_iv_manipulation_detailed",
        "methods_t2_procedure_full_chronological",
        "methods_t2_design_coherent_with_rq_and_hypotheses",
        # Tier 2 additional
        "methods_t2a_conditions_compared_explicitly",
        "methods_t2a_reliability_validity_one_questionnaire",
        "methods_t2a_allocation_method_described",
        "methods_t2a_rewards_mentioned",
        # Tier 3 core
        "methods_t3_conditions_unambiguous",
        "methods_t3_all_questionnaires_complete_with_cutoffs",
        "methods_t3_procedure_complete_and_clean",
        "methods_t3_information_in_right_place",
        # Tier 3 additional
        "methods_t3a_reliability_validity_all_instruments",
        "methods_t3a_design_choices_argued",
        "methods_t3a_standardization_described",
        # Tier 4 core
        "methods_t4_complete_correct_clearly_written",
        "methods_t4_every_choice_traceable_to_rq",
        # Tier 4 additional
        "methods_t4a_goes_beyond_checklist",
        "methods_t4a_unusually_thorough",
    ],
    "results": [
        # Tier 0
        "results_t0_no_statistical_analysis",
        "results_t0_analysis_unrelated_to_design",
        "results_t0_construct_level_only",
        # Tier 1 core
        "results_t1_correct_test_applied",
        "results_t1_main_result_with_statistic_and_p",
        "results_t1_descriptive_stats_reported",
        "results_t1_operational_level_reporting",
        # Tier 1 additional
        "results_t1a_outlier_check_mentioned",
        "results_t1a_assumption_check_reported",
        "results_t1a_table_or_figure_included",
        # Tier 2 core
        "results_t2_all_effects_reported",
        "results_t2_demographic_table_complete",
        "results_t2_means_sd_table_complete",
        "results_t2_graph_included",
        "results_t2_assumption_checks_reported",
        "results_t2_outlier_check_with_decision",
        "results_t2_posthoc_with_correction",
        # Tier 2 additional
        "results_t2a_effect_sizes_reported",
        "results_t2a_tables_figures_correctly_formatted",
        "results_t2a_clt_invoked_if_needed",
        # Tier 3 core
        "results_t3_assumption_checks_complete",
        "results_t3_posthoc_complete_and_correct",
        "results_t3_tables_figures_labelled",
        "results_t3_coherent_with_design",
        # Tier 3 additional
        "results_t3a_effect_sizes_all_comparisons",
        "results_t3a_violations_handled_and_justified",
        "results_t3a_follows_sensible_structure",
        # Tier 4 core
        "results_t4_complete_correct_clearly_written",
        "results_t4_fully_consistent_apa_and_sample",
        # Tier 4 additional
        "results_t4a_statistical_reasoning_demonstrated",
        "results_t4a_exceptionally_well_organized",
    ],
    "discussion": [
        # Tier 0
        "discussion_t0_no_main_conclusion",
        "discussion_t0_no_construct_level_interpretation",
        "discussion_t0_disconnected_from_introduction",
        # Tier 1 core
        "discussion_t1_rq_revisited_and_conclusion_stated",
        "discussion_t1_hypothesis_addressed",
        "discussion_t1_main_finding_interpreted",
        "discussion_t1_limitation_mentioned",
        "discussion_t1_future_direction_mentioned",
        # Tier 1 additional
        "discussion_t1a_compared_to_prior_study",
        "discussion_t1a_broader_implication_mentioned",
        # Tier 2 core
        "discussion_t2_all_findings_interpreted",
        "discussion_t2_compared_to_prior_with_conclusion",
        "discussion_t2_unexpected_finding_discussed",
        "discussion_t2_theoretical_framework_revisited",
        "discussion_t2_limitations_specific",
        "discussion_t2_broader_implication_argued",
        # Tier 2 additional
        "discussion_t2a_explanations_grounded_in_research",
        "discussion_t2a_future_research_from_limitations",
        "discussion_t2a_clearly_structured",
        # Tier 3 core
        "discussion_t3_reasoning_complete_and_coherent",
        "discussion_t3_comparisons_substantive",
        "discussion_t3_framework_evaluated",
        "discussion_t3_limitations_linked_to_conclusions",
        "discussion_t3_future_research_specific",
        # Tier 3 additional
        "discussion_t3a_alternative_interpretations",
        "discussion_t3a_connects_to_societal_relevance",
        "discussion_t3a_broader_research_context",
        # Tier 4 core
        "discussion_t4_reasoning_very_coherent_and_complete",
        "discussion_t4_reads_as_unified_argument",
        # Tier 4 additional
        "discussion_t4a_novel_insights",
        "discussion_t4a_alternative_interpretations_evaluated",
        "discussion_t4a_mature_critical_thinking",
    ],
    "lang_style": [
        # Tier 0
        "lang_t0_broadly_unscientific_or_unreadable",
        "lang_t0_paragraphs_unstructured",
        "lang_t0_no_references",
        # Tier 1 core
        "lang_t1_generally_understandable",
        "lang_t1_paragraphs_have_focus",
        "lang_t1_scientific_language_used",
        # Tier 1 additional
        "lang_t1a_topical_sentences_present",
        "lang_t1a_errors_occasional",
        "lang_t1a_apa_mostly_correct",
        "lang_t1a_structure_mostly_coherent",
        # Tier 2 core
        "lang_t2_sections_clearly_structured",
        "lang_t2_scientific_language_correct",
        "lang_t2_few_grammar_errors",
        "lang_t2_few_apa_errors",
        # Tier 2 additional
        "lang_t2a_sentences_concise",
        "lang_t2a_smooth_transitions",
        "lang_t2a_vague_statements_rare",
        # Tier 3 core
        "lang_t3_all_paragraphs_single_focus",
        "lang_t3_scientific_language_near_faultless",
        "lang_t3_errors_rare",
        # Tier 3 additional
        "lang_t3a_minimal_redundancy",
        "lang_t3a_clear_precise_easy_to_follow",
        "lang_t3a_topical_sentences_all_paragraphs",
        # Tier 4 core
        "lang_t4_graceful_mature_style",
        "lang_t4_no_apa_errors",
        "lang_t4_virtually_no_redundancy",
        # Tier 4 additional
        "lang_t4a_style_enhances_argumentation",
        "lang_t4a_polished_cohesive_document",
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
        "reasoning_summary",
    ]
    for sec in SECTION_KEYS:
        if include_checklist:
            base.extend(CHECKLIST[sec])
        if include_feedback:
            base.append(f"{sec}_feedback")
        base.append(f"{sec}_grade")
    base += ["final_grade", "missing_sections"]
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
