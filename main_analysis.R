options(mc.cores = parallel::detectCores())
options(brms.backend = "cmdstanr")

library(tidyverse)
library(readxl)
library(BayesFactor)
library(brms)
library(cmdstanr)
library(psych)
library(cocor)
library(flextable)
library(officer)
library(viridis)
library(patchwork)


# Data preparation --------------------------------------------------------


read_regrader_id <- function(path) {
  read_excel(path) %>%
    filter(str_detect(student_id, "^\\d+$")) %>%
    mutate(
      student_id = as.integer(student_id),
      regrader_id = as.integer(regrader_id),
      introduction = as.numeric(introduction)
    ) %>%
    filter(!is.na(introduction)) %>%
    select(student_id, regrader_id)
}

regrader_ids <- map(
  c("data/Regrading/grader-1_grading.xlsx",
    "data/Regrading/grader-2_grading.xlsx",
    "data/Regrading/grader-3_grading.xlsx",
    "data/Regrading/grader-4_grading.xlsx",
    "data/Regrading/grader-5_grading.xlsx"),
  read_regrader_id
) %>%
  bind_rows() %>%
  distinct(student_id, .keep_all = TRUE)

regrade_cols <- c("regrade_intro", "regrade_methods", "regrade_results",
                  "regrade_discussion", "regrade_lang_style", "regrade_final")

read_csv("outputs/run-28/combined_with_regrades.csv") %>%
  left_join(regrader_ids, by = "student_id") %>%
  mutate(across(all_of(regrade_cols),
                ~if_else(pipeline_id == "improved", NA_real_, .))) %>%
  write_csv("outputs/run-28/final_data.csv")

luna_data <- read_csv("outputs/run-28/final_data.csv")

nano_data <- read_csv("outputs/run-25/results.csv") %>%
  left_join(
    luna_data %>%
      distinct(student_id, .keep_all = TRUE) %>%
      select(student_id, all_of(regrade_cols), regrader_id,
             Grade, Language, PseudoGroup, groupid.pseudoTutorID,
             Tutorial_Language, missing_sections),
    by = "student_id"
  ) %>%
  mutate(
    across(all_of(regrade_cols),
           ~if_else(pipeline_id == "improved", NA_real_, .)),
    Grade = NA_real_,
    run = as.integer(str_extract(repetition, "\\d+"))
  )

data <- bind_rows(luna_data, nano_data) %>%
  rename(rubric = pipeline_id)

student_cols <- c("student_id", "missing_sections", "Language",
                  "PseudoGroup", "groupid.pseudoTutorID", "Tutorial_Language")
llm_meta_cols <- c("Unnamed: 0", "run_id", "run_label", "rubric",
                   "repetition", "model", "reasoning_effort", "timestamp",
                   "input_tokens", "output_tokens", "run")
llm_rubric_cols <- c("reasoning_summary",
                     names(data)[which(names(data) == "intro_t0_no_rq"):
                                   which(names(data) == "final_grade")])

long_data <- bind_rows(
  data %>%
    select(all_of(student_cols), all_of(llm_meta_cols), all_of(llm_rubric_cols)) %>%
    rename(rater = model),
  data %>%
    filter(model == "gpt-5.6-luna") %>%
    distinct(student_id, .keep_all = TRUE) %>%
    select(all_of(student_cols), Grade, groupid.pseudoTutorID) %>%
    mutate(
      rater = "tutor",
      rubric = NA_character_,
      repetition = NA_character_,
      reasoning_effort = NA_character_,
      intro_grade = NA_real_,
      methods_grade = NA_real_,
      results_grade = NA_real_,
      discussion_grade = NA_real_,
      lang_style_grade = NA_real_,
      final_grade = Grade,
      tutor_id = as.character(groupid.pseudoTutorID)
    ) %>%
    select(-Grade),
  data %>%
    filter(model == "gpt-5.6-luna") %>%
    distinct(student_id, .keep_all = TRUE) %>%
    select(all_of(student_cols),
           intro_grade = regrade_intro,
           methods_grade = regrade_methods,
           results_grade = regrade_results,
           discussion_grade = regrade_discussion,
           lang_style_grade = regrade_lang_style,
           final_grade = regrade_final,
           human_rater_id = regrader_id) %>%
    mutate(
      rater = "regrader",
      rubric = NA_character_,
      repetition = NA_character_,
      reasoning_effort = NA_character_,
      regrader_id = as.character(human_rater_id)
    )
) %>%
  select(-any_of("Grade"))

write_csv(long_data, "outputs/run-final/long_data.csv")

long <- read_csv("outputs/run-final/long_data.csv")

out <- "outputs/run-final/"

section_cols <- c("intro_grade", "methods_grade", "results_grade",
                  "discussion_grade", "lang_style_grade")
section_labels <- c("Introduction", "Methods", "Results",
                    "Discussion", "Language & Style")

rater_levels <- c("tutor", "regrade",
                  "llm_gpt-5.6-luna_original", "llm_gpt-5.6-luna_improved",
                  "llm_gpt-5-nano_original", "llm_gpt-5-nano_improved")
rater_labels <- c("Tutor", "Regrader",
                  "Luna original", "Luna improved",
                  "Nano original", "Nano improved")


# Main analyses -----------------------------------------------------------


set.seed(5)

all_grades <- long %>%
  filter(rater == "tutor") %>%
  select(student_id, final_grade) %>%
  rename(tutor = final_grade) %>%
  left_join(
    long %>%
      filter(rater == "regrader") %>%
      select(student_id, final_grade) %>%
      rename(regrade = final_grade),
    by = "student_id"
  ) %>%
  left_join(
    long %>%
      filter(!rater %in% c("tutor", "regrader")) %>%
      group_by(student_id, rater, rubric) %>%
      summarise(llm_mean = mean(final_grade, na.rm = TRUE), .groups = "drop") %>%
      pivot_wider(names_from = c(rater, rubric), values_from = llm_mean,
                  names_glue = "llm_{rater}_{rubric}"),
    by = "student_id"
  )

# Descriptives

desc_final <- all_grades %>%
  pivot_longer(-student_id, names_to = "rater", values_to = "grade") %>%
  drop_na() %>%
  group_by(rater) %>%
  summarise(n = n(),
            mean = mean(grade),
            sd = sd(grade),
            min = min(grade),
            max = max(grade),
            .groups = "drop")

write_csv(desc_final, paste0(out, "descriptives.csv"))

# Within-pipeline consistency

consistency <- long %>%
  filter(!rater %in% c("tutor", "regrader")) %>%
  select(student_id, rater, rubric, repetition, final_grade, all_of(section_cols)) %>%
  pivot_longer(c(final_grade, all_of(section_cols)),
               names_to = "grade_type", values_to = "grade") %>%
  group_by(student_id, rater, rubric, grade_type) %>%
  summarise(rep_sd = sd(grade, na.rm = TRUE), .groups = "drop") %>%
  group_by(rater, rubric, grade_type) %>%
  summarise(mean_sd = mean(rep_sd, na.rm = TRUE),
            sd_sd = sd(rep_sd, na.rm = TRUE),
            .groups = "drop")

write_csv(consistency, paste0(out, "consistency.csv"))

icc_consistency <- long %>%
  filter(!rater %in% c("tutor", "regrader")) %>%
  select(student_id, rater, rubric, repetition, final_grade, all_of(section_cols)) %>%
  pivot_longer(c(final_grade, all_of(section_cols)),
               names_to = "grade_type", values_to = "grade") %>%
  pivot_wider(names_from = repetition, values_from = grade,
              names_prefix = "rep_") %>%
  group_by(rater, rubric, grade_type) %>%
  group_map(~{
    mat <- select(.x, starts_with("rep_")) %>% drop_na()
    res <- ICC(mat, missing = FALSE, alpha = 0.05)$results %>%
      filter(type == "ICC2k")
    tibble(rater = .y$rater,
           rubric = .y$rubric,
           grade_type = .y$grade_type,
           icc = res$ICC,
           ci_low = res$`lower bound`,
           ci_high = res$`upper bound`)
  }) %>%
  bind_rows()

write_csv(icc_consistency, paste0(out, "icc_consistency.csv"))

# Assumption checks

ba_data_raw <- bind_rows(
  all_grades %>% transmute(student_id, x = tutor, y = regrade,
                           comparison = "tutor vs regrader"),
  all_grades %>% transmute(student_id, x = tutor, y = `llm_gpt-5.6-luna_original`,
                           comparison = "tutor vs luna original"),
  all_grades %>% transmute(student_id, x = tutor, y = `llm_gpt-5.6-luna_improved`,
                           comparison = "tutor vs luna improved"),
  all_grades %>% transmute(student_id, x = tutor, y = `llm_gpt-5-nano_original`,
                           comparison = "tutor vs nano original"),
  all_grades %>% transmute(student_id, x = tutor, y = `llm_gpt-5-nano_improved`,
                           comparison = "tutor vs nano improved"),
  all_grades %>% transmute(student_id, x = regrade, y = `llm_gpt-5.6-luna_original`,
                           comparison = "regrader vs luna original"),
  all_grades %>% transmute(student_id, x = regrade, y = `llm_gpt-5.6-luna_improved`,
                           comparison = "regrader vs luna improved"),
  all_grades %>% transmute(student_id, x = regrade, y = `llm_gpt-5-nano_original`,
                           comparison = "regrader vs nano original"),
  all_grades %>% transmute(student_id, x = regrade, y = `llm_gpt-5-nano_improved`,
                           comparison = "regrader vs nano improved")
) %>%
  drop_na() %>%
  mutate(mean_pair = (x + y) / 2, diff = y - x)

ba_shapiro <- ba_data_raw %>%
  group_by(comparison) %>%
  summarise(W = shapiro.test(diff)$statistic,
            p = shapiro.test(diff)$p.value,
            .groups = "drop")

write_csv(ba_shapiro, paste0(out, "assumption_ba_shapiro.csv"))

ba_qq <- ba_data_raw %>%
  ggplot(aes(sample = diff)) +
  stat_qq() + stat_qq_line() +
  facet_wrap(~comparison, ncol = 3) +
  labs(title = "Q-Q plots: Bland-Altman differences") +
  theme(strip.background = element_blank())

ggsave(paste0(out, "assumption_ba_qq.png"), ba_qq,
       width = 22, height = 18, units = "cm", dpi = 300)

grades_shapiro <- all_grades %>%
  select(-student_id) %>%
  pivot_longer(everything(), names_to = "rater", values_to = "grade") %>%
  drop_na() %>%
  group_by(rater) %>%
  summarise(W = shapiro.test(grade)$statistic,
            p = shapiro.test(grade)$p.value,
            .groups = "drop")

write_csv(grades_shapiro, paste0(out, "assumption_grades_shapiro.csv"))

cor_pairs <- list(
  c("tutor", "regrade", "tutor vs regrader"),
  c("tutor", "llm_gpt-5.6-luna_original", "tutor vs luna original"),
  c("tutor", "llm_gpt-5.6-luna_improved", "tutor vs luna improved"),
  c("tutor", "llm_gpt-5-nano_original", "tutor vs nano original"),
  c("tutor", "llm_gpt-5-nano_improved", "tutor vs nano improved"),
  c("regrade", "llm_gpt-5.6-luna_original", "regrader vs luna original"),
  c("regrade", "llm_gpt-5.6-luna_improved", "regrader vs luna improved"),
  c("regrade", "llm_gpt-5-nano_original", "regrader vs nano original"),
  c("regrade", "llm_gpt-5-nano_improved", "regrader vs nano improved"),
  c("llm_gpt-5.6-luna_original", "llm_gpt-5.6-luna_improved", "luna: original vs improved"),
  c("llm_gpt-5-nano_original", "llm_gpt-5-nano_improved", "nano: original vs improved")
)

cor_scatter <- wrap_plots(
  map(cor_pairs, function(p) {
    all_grades %>%
      select(x = all_of(p[1]), y = all_of(p[2])) %>%
      drop_na() %>%
      ggplot(aes(x, y)) +
      geom_abline(intercept = 0, slope = 1,
                  linetype = "dashed", color = "grey60", linewidth = 0.5) +
      geom_point(shape = 1) +
      geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, color = "grey40") +
      scale_x_continuous(limits = c(4.5, 10), breaks = seq(5, 10, 2)) +
      scale_y_continuous(limits = c(4.5, 10), breaks = seq(5, 10, 2)) +
      coord_fixed() +
      labs(title = p[3], x = p[1], y = p[2])
  }),
  ncol = 3
)

ggsave(paste0(out, "assumption_cor_scatter.png"), cor_scatter,
       width = 22, height = 26, units = "cm", dpi = 300)

# Bland-Altman analysis

ba_summary <- ba_data_raw %>%
  group_by(comparison) %>%
  summarise(
    mean_diff = mean(diff),
    sd_diff = sd(diff),
    loa_lower = mean_diff - 1.96 * sd_diff,
    loa_upper = mean_diff + 1.96 * sd_diff,
    .groups = "drop"
  )

write_csv(ba_summary, paste0(out, "ba_summary.csv"))

bias_ttests <- ba_data_raw %>%
  group_by(comparison) %>%
  group_map(~{
    bf <- ttestBF(x = .x$diff, mu = 0)
    posts <- posterior(bf, iterations = 10000) %>% as.data.frame()
    tibble(
      comparison = .y$comparison,
      mean_diff = mean(.x$diff),
      sd_diff = sd(.x$diff),
      delta_median = median(posts$delta),
      ci_low = quantile(posts$delta, 0.025),
      ci_high = quantile(posts$delta, 0.975),
      BF10 = exp(bf@bayesFactor$bf)
    )
  }) %>%
  bind_rows()

write_csv(bias_ttests, paste0(out, "bias_ttests.csv"))

ba_bias_fits <- map(unique(ba_data_raw$comparison), function(cmp) {
  brm(
    diff ~ mean_pair,
    data = filter(ba_data_raw, comparison == cmp),
    prior = c(prior(normal(0, 2), class = Intercept),
              prior(normal(0, 1), class = b),
              prior(exponential(1), class = sigma)),
    chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
  )
}) %>% set_names(unique(ba_data_raw$comparison))

ba_bias_ppcheck <- wrap_plots(
  imap(ba_bias_fits, ~pp_check(.x, ndraws = 100) + labs(title = .y)),
  ncol = 3
)
ggsave(paste0(out, "assumption_bias_ppcheck.png"), ba_bias_ppcheck,
       width = 22, height = 26, units = "cm", dpi = 300)

ba_bias_resid <- wrap_plots(
  imap(ba_bias_fits, function(fit, cmp) {
    tibble(fitted = fitted(fit)[, "Estimate"],
           residual = residuals(fit)[, "Estimate"]) %>%
      ggplot(aes(fitted, residual)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
      geom_point(shape = 1) +
      labs(title = cmp, x = "Fitted", y = "Residual")
  }),
  ncol = 3
)
ggsave(paste0(out, "assumption_bias_residuals.png"), ba_bias_resid,
       width = 22, height = 26, units = "cm", dpi = 300)

ba_bias <- imap_dfr(ba_bias_fits, function(fit, cmp) {
  b <- as_draws_df(fit)$b_mean_pair
  tibble(comparison = cmp,
         slope_median = median(b),
         ci_low = quantile(b, 0.025),
         ci_high = quantile(b, 0.975),
         p_neg = mean(b < 0),
         p_pos = mean(b > 0))
})

write_csv(ba_bias, paste0(out, "proportional_bias.csv"))

# Bayesian correlations

cor_results <- map_dfr(cor_pairs, function(p) {
  d <- all_grades %>% select(x = all_of(p[1]), y = all_of(p[2])) %>% drop_na()
  bf <- correlationBF(d$x, d$y)
  posts <- posterior(bf, iterations = 10000) %>% as.data.frame()
  tibble(comparison = p[3],
         n = nrow(d),
         rho_median = median(posts$rho),
         ci_low = quantile(posts$rho, 0.025),
         ci_high = quantile(posts$rho, 0.975),
         BF10 = exp(bf@bayesFactor$bf),
         p_pos = mean(posts$rho > 0))
})

write_csv(cor_results, paste0(out, "correlations.csv"))

fisher_z_compare <- function(label, col_x, col_a, col_b) {
  d_a <- all_grades %>% select(x = all_of(col_x), y = all_of(col_a)) %>% drop_na()
  d_b <- all_grades %>% select(x = all_of(col_x), y = all_of(col_b)) %>% drop_na()
  r_a <- cor(d_a$x, d_a$y)
  r_b <- cor(d_b$x, d_b$y)
  n_a <- nrow(d_a)
  n_b <- nrow(d_b)
  z_a <- 0.5 * log((1 + r_a) / (1 - r_a))
  z_b <- 0.5 * log((1 + r_b) / (1 - r_b))
  se <- sqrt(1 / (n_a - 3) + 1 / (n_b - 3))
  z_diff <- z_a - z_b
  ci_low <- z_diff - 1.96 * se
  ci_high <- z_diff + 1.96 * se
  p_val <- 2 * pnorm(-abs(z_diff / se))
  tibble(comparison = label,
         r_a = round(r_a, 3),
         r_b = round(r_b, 3),
         z_diff = round(z_diff, 3),
         ci_low = round(ci_low, 3),
         ci_high = round(ci_high, 3),
         p = round(p_val, 4))
}

fisher_comparisons <- bind_rows(
  fisher_z_compare("tutor: luna improved vs original",
                   "tutor", "llm_gpt-5.6-luna_improved", "llm_gpt-5.6-luna_original"),
  fisher_z_compare("tutor: nano improved vs original",
                   "tutor", "llm_gpt-5-nano_improved", "llm_gpt-5-nano_original"),
  fisher_z_compare("regrader: luna improved vs original",
                   "regrade", "llm_gpt-5.6-luna_improved", "llm_gpt-5.6-luna_original"),
  fisher_z_compare("regrader: nano improved vs original",
                   "regrade", "llm_gpt-5-nano_improved", "llm_gpt-5-nano_original"),
  fisher_z_compare("tutor: luna vs nano (original rubric)",
                   "tutor", "llm_gpt-5.6-luna_original", "llm_gpt-5-nano_original"),
  fisher_z_compare("tutor: luna vs nano (improved rubric)",
                   "tutor", "llm_gpt-5.6-luna_improved", "llm_gpt-5-nano_improved"),
  fisher_z_compare("regrader: luna vs nano (original rubric)",
                   "regrade", "llm_gpt-5.6-luna_original", "llm_gpt-5-nano_original"),
  fisher_z_compare("regrader: luna vs nano (improved rubric)",
                   "regrade", "llm_gpt-5.6-luna_improved", "llm_gpt-5-nano_improved")
)

write_csv(fisher_comparisons, paste0(out, "fisher_z_comparisons.csv"))

# Agreement model

mlm_data <- long %>%
  filter(!rater %in% c("tutor", "regrader")) %>%
  group_by(student_id, rater, rubric) %>%
  summarise(llm_grade = mean(final_grade, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    model_type = factor(if_else(rater == "gpt-5.6-luna", "luna", "nano"),
                        levels = c("nano", "luna")),
    rubric_type = factor(rubric, levels = c("original", "improved"))
  ) %>%
  left_join(all_grades %>% select(student_id, tutor, regrade), by = "student_id")

agreement_model_tutor <- brm(
  llm_grade ~ tutor * model_type * rubric_type + (1 | student_id),
  data = mlm_data,
  prior = c(prior(normal(7, 2), class = Intercept),
            prior(normal(0, 1), class = b),
            prior(exponential(1), class = sd),
            prior(exponential(1), class = sigma)),
  chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
)

ggsave(paste0(out, "assumption_agreement_tutor_ppcheck.png"),
       pp_check(agreement_model_tutor, ndraws = 100),
       width = 17.8, height = 10, units = "cm", dpi = 300)

agreement_tutor_ranef_qq <- ranef(agreement_model_tutor)$student_id[,, "Intercept"] %>%
  as.data.frame() %>%
  ggplot(aes(sample = Estimate)) +
  stat_qq() + stat_qq_line() +
  labs(title = "Random intercepts Q-Q: agreement model (tutor)")
ggsave(paste0(out, "assumption_agreement_tutor_ranef_qq.png"),
       agreement_tutor_ranef_qq,
       width = 17.8, height = 10, units = "cm", dpi = 300)

agreement_model_regrader <- brm(
  llm_grade ~ regrade * model_type * rubric_type + (1 | student_id),
  data = mlm_data %>% drop_na(regrade),
  prior = c(prior(normal(7, 2), class = Intercept),
            prior(normal(0, 1), class = b),
            prior(exponential(1), class = sd),
            prior(exponential(1), class = sigma)),
  chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
)

ggsave(paste0(out, "assumption_agreement_regrader_ppcheck.png"),
       pp_check(agreement_model_regrader, ndraws = 100),
       width = 17.8, height = 10, units = "cm", dpi = 300)

agreement_regrader_ranef_qq <- ranef(agreement_model_regrader)$student_id[,, "Intercept"] %>%
  as.data.frame() %>%
  ggplot(aes(sample = Estimate)) +
  stat_qq() + stat_qq_line() +
  labs(title = "Random intercepts Q-Q: agreement model (regrader)")
ggsave(paste0(out, "assumption_agreement_regrader_ranef_qq.png"),
       agreement_regrader_ranef_qq,
       width = 17.8, height = 10, units = "cm", dpi = 300)

extract_agreement_summary <- function(fit, human_var) {
  as_draws_df(fit) %>%
    select(starts_with("b_")) %>%
    pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
    group_by(parameter) %>%
    summarise(
      median = median(value),
      ci_low = quantile(value, 0.025),
      ci_high = quantile(value, 0.975),
      p_pos = mean(value > 0),
      p_neg = mean(value < 0),
      .groups = "drop"
    ) %>%
    mutate(human_criterion = human_var)
}

agreement_summary <- bind_rows(
  extract_agreement_summary(agreement_model_tutor, "tutor"),
  extract_agreement_summary(agreement_model_regrader, "regrader")
)

write_csv(agreement_summary, paste0(out, "agreement_model_summary.csv"))

# Variance decomposition

var_data_1a <- bind_rows(
  all_grades %>%
    select(student_id, tutor, regrade) %>%
    pivot_longer(c(tutor, regrade),
                 names_to = "rater_type", values_to = "final_grade"),
  long %>%
    filter(!rater %in% c("tutor", "regrader")) %>%
    group_by(student_id, rater, rubric) %>%
    summarise(final_grade = mean(final_grade, na.rm = TRUE), .groups = "drop") %>%
    mutate(rater_type = if_else(rater == "gpt-5.6-luna", "luna", "nano")) %>%
    select(student_id, rater_type, final_grade)
) %>%
  drop_na(final_grade) %>%
  mutate(
    rater_type = factor(rater_type, levels = c("tutor", "regrade", "luna", "nano")),
    student_id = factor(student_id)
  )

model_1a <- brm(
  final_grade ~ rater_type + (1 | student_id),
  data = var_data_1a,
  prior = c(prior(normal(7, 2), class = Intercept),
            prior(normal(0, 1), class = b),
            prior(exponential(1), class = sd),
            prior(exponential(1), class = sigma)),
  chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
)

ggsave(paste0(out, "assumption_model1a_ppcheck.png"),
       pp_check(model_1a, ndraws = 100),
       width = 17.8, height = 10, units = "cm", dpi = 300)

model1a_ranef_qq <- ranef(model_1a)$student_id[,, "Intercept"] %>%
  as.data.frame() %>%
  ggplot(aes(sample = Estimate)) +
  stat_qq() + stat_qq_line() +
  labs(title = "Random intercepts Q-Q: model 1a")
ggsave(paste0(out, "assumption_model1a_ranef_qq.png"), model1a_ranef_qq,
       width = 17.8, height = 10, units = "cm", dpi = 300)

extract_var_components <- function(fit, label) {
  as_draws_df(fit) %>%
    transmute(
      sd_student = `sd_student_id__Intercept`,
      sigma = sigma,
      var_student = sd_student^2,
      var_resid = sigma^2,
      icc_student = var_student / (var_student + var_resid)
    ) %>%
    summarise(across(everything(),
                     list(median = median,
                          ci_low = ~quantile(.x, 0.025),
                          ci_high = ~quantile(.x, 0.975)))) %>%
    mutate(model = label)
}

var_data_1b <- long %>%
  filter(!rater %in% c("tutor", "regrader"), rubric == "original") %>%
  group_by(student_id, rater) %>%
  summarise(final_grade = mean(final_grade, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    model_type = factor(if_else(rater == "gpt-5.6-luna", "luna", "nano"),
                        levels = c("nano", "luna")),
    student_id = factor(student_id)
  ) %>%
  drop_na(final_grade)

model_1b <- brm(
  final_grade ~ model_type + (1 | student_id),
  data = var_data_1b,
  prior = c(prior(normal(7, 2), class = Intercept),
            prior(normal(0, 1), class = b),
            prior(exponential(1), class = sd),
            prior(exponential(1), class = sigma)),
  chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
)

ggsave(paste0(out, "assumption_model1b_ppcheck.png"),
       pp_check(model_1b, ndraws = 100),
       width = 17.8, height = 10, units = "cm", dpi = 300)

var_data_1c <- long %>%
  filter(!rater %in% c("tutor", "regrader"), rubric == "improved") %>%
  group_by(student_id, rater) %>%
  summarise(final_grade = mean(final_grade, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    model_type = factor(if_else(rater == "gpt-5.6-luna", "luna", "nano"),
                        levels = c("nano", "luna")),
    student_id = factor(student_id)
  ) %>%
  drop_na(final_grade)

model_1c <- brm(
  final_grade ~ model_type + (1 | student_id),
  data = var_data_1c,
  prior = c(prior(normal(7, 2), class = Intercept),
            prior(normal(0, 1), class = b),
            prior(exponential(1), class = sd),
            prior(exponential(1), class = sigma)),
  chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
)

ggsave(paste0(out, "assumption_model1c_ppcheck.png"),
       pp_check(model_1c, ndraws = 100),
       width = 17.8, height = 10, units = "cm", dpi = 300)

var_data_1d <- long %>%
  filter(!rater %in% c("tutor", "regrader")) %>%
  group_by(student_id, rater, rubric) %>%
  summarise(final_grade = mean(final_grade, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    model_type = factor(if_else(rater == "gpt-5.6-luna", "luna", "nano"),
                        levels = c("nano", "luna")),
    rubric_type = factor(rubric, levels = c("original", "improved")),
    student_id = factor(student_id)
  ) %>%
  drop_na(final_grade)

model_1d <- brm(
  final_grade ~ model_type + rubric_type + (1 | student_id),
  data = var_data_1d,
  prior = c(prior(normal(7, 2), class = Intercept),
            prior(normal(0, 1), class = b),
            prior(exponential(1), class = sd),
            prior(exponential(1), class = sigma)),
  chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
)

ggsave(paste0(out, "assumption_model1d_ppcheck.png"),
       pp_check(model_1d, ndraws = 100),
       width = 17.8, height = 10, units = "cm", dpi = 300)

var_components_all <- bind_rows(
  extract_var_components(model_1a, "1a: all raters"),
  extract_var_components(model_1b, "1b: LLM original rubric"),
  extract_var_components(model_1c, "1c: LLM improved rubric"),
  extract_var_components(model_1d, "1d: LLM both rubrics")
)

write_csv(var_components_all, paste0(out, "variance_components.csv"))

extract_fixed <- function(fit, label) {
  as_draws_df(fit) %>%
    select(starts_with("b_")) %>%
    pivot_longer(everything(), names_to = "parameter", values_to = "value") %>%
    group_by(parameter) %>%
    summarise(
      median = median(value),
      ci_low = quantile(value, 0.025),
      ci_high = quantile(value, 0.975),
      p_pos = mean(value > 0),
      p_neg = mean(value < 0),
      .groups = "drop"
    ) %>%
    mutate(model = label)
}

var_fixed_effects <- bind_rows(
  extract_fixed(model_1a, "1a: all raters"),
  extract_fixed(model_1d, "1d: LLM both rubrics")
)

write_csv(var_fixed_effects, paste0(out, "variance_fixed_effects.csv"))

# Human-rater variance

regrader_var_data <- long %>%
  filter(rater == "regrader") %>%
  select(student_id, regrader_id, final_grade) %>%
  drop_na() %>%
  mutate(student_id = factor(student_id),
         regrader_id = factor(regrader_id))

model_3a <- brm(
  final_grade ~ (1 | regrader_id),
  data = regrader_var_data,
  prior = c(prior(normal(7, 2), class = Intercept),
            prior(exponential(1), class = sd),
            prior(exponential(1), class = sigma)),
  chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
)

regrader_icc <- as_draws_df(model_3a) %>%
  transmute(
    sd_regrader = `sd_regrader_id__Intercept`,
    sigma = sigma,
    var_regrader = sd_regrader^2,
    var_resid = sigma^2,
    icc_regrader = var_regrader / (var_regrader + var_resid)
  ) %>%
  summarise(across(everything(),
                   list(median = median,
                        ci_low = ~quantile(.x, 0.025),
                        ci_high = ~quantile(.x, 0.975)))) %>%
  mutate(model = "3a: regrader")

tutor_var_data <- long %>%
  filter(rater == "tutor") %>%
  select(student_id, tutor_id, final_grade) %>%
  drop_na() %>%
  mutate(student_id = factor(student_id),
         tutor_id = factor(tutor_id))

model_3b <- brm(
  final_grade ~ (1 | tutor_id),
  data = tutor_var_data,
  prior = c(prior(normal(7, 2), class = Intercept),
            prior(exponential(1), class = sd),
            prior(exponential(1), class = sigma)),
  chains = 4, iter = 4000, warmup = 1000, seed = 5, silent = 2
)

tutor_icc <- as_draws_df(model_3b) %>%
  transmute(
    sd_tutor = `sd_tutor_id__Intercept`,
    sigma = sigma,
    var_tutor = sd_tutor^2,
    var_resid = sigma^2,
    icc_tutor = var_tutor / (var_tutor + var_resid)
  ) %>%
  summarise(across(everything(),
                   list(median = median,
                        ci_low = ~quantile(.x, 0.025),
                        ci_high = ~quantile(.x, 0.975)))) %>%
  mutate(model = "3b: tutor (sparse — interpret cautiously)")

human_rater_icc <- bind_rows(regrader_icc, tutor_icc)
write_csv(human_rater_icc, paste0(out, "human_rater_icc.csv"))

# Section-level analysis

section_long <- long %>%
  select(student_id, rater, rubric, all_of(section_cols)) %>%
  pivot_longer(all_of(section_cols),
               names_to = "section", values_to = "grade") %>%
  mutate(section = factor(section, levels = section_cols, labels = section_labels))

section_means <- section_long %>%
  group_by(rater, rubric, section) %>%
  summarise(mean_grade = mean(grade, na.rm = TRUE),
            sd_grade = sd(grade, na.rm = TRUE),
            .groups = "drop")

write_csv(section_means, paste0(out, "section_means.csv"))

section_diffs <- section_long %>%
  filter(rater == "regrader") %>%
  select(student_id, section, regrade_grade = grade) %>%
  left_join(
    section_long %>%
      filter(!rater %in% c("tutor", "regrader")) %>%
      group_by(student_id, rater, rubric, section) %>%
      summarise(llm_grade = mean(grade, na.rm = TRUE), .groups = "drop"),
    by = c("student_id", "section")
  ) %>%
  group_by(rater, rubric, section) %>%
  summarise(mean_diff = mean(llm_grade - regrade_grade, na.rm = TRUE),
            .groups = "drop")

write_csv(section_diffs, paste0(out, "section_differences.csv"))

section_sd_ratio <- section_means %>%
  filter(rater == "regrader") %>%
  select(section, sd_regrade = sd_grade) %>%
  left_join(
    section_means %>%
      filter(!rater %in% c("tutor", "regrader")) %>%
      select(rater, rubric, section, sd_llm = sd_grade),
    by = "section"
  ) %>%
  mutate(sd_ratio = sd_regrade / sd_llm)

write_csv(section_sd_ratio, paste0(out, "section_sd_ratios.csv"))


# Tables and figures ------------------------------------------------------


set_flextable_defaults(font.family = "Times New Roman", font.size = 12)
theme_set(theme_classic(base_family = "Times New Roman", base_size = 12))

apa_table <- function(ft, note = NULL) {
  ft <- ft %>%
    border_remove() %>%
    hline_top(border = fp_border(width = 1), part = "head") %>%
    hline_bottom(border = fp_border(width = 1), part = "head") %>%
    hline_bottom(border = fp_border(width = 1), part = "body") %>%
    align(align = "left", part = "all") %>%
    align(j = -1, align = "center", part = "all") %>%
    padding(padding = 2, part = "all")
  if (!is.null(note)) ft <- add_footer_lines(ft, paste("Note.", note))
  ft
}

# Table 1: Descriptives
desc_final %>%
  mutate(rater = factor(rater, levels = rater_levels, labels = rater_labels),
         across(where(is.numeric), ~round(.x, 2))) %>%
  arrange(rater) %>%
  rename(Rater = rater, N = n, Mean = mean, SD = sd,
         Minimum = min, Maximum = max) %>%
  flextable() %>%
  apa_table(note = "LLM grades represent the mean across three runs per pipeline.") %>%
  save_as_docx(path = paste0(out, "table1_descriptives.docx"))

# Table 2: Within-pipeline consistency
left_join(
  read_csv(paste0(out, "icc_consistency.csv")) %>%
    mutate(CI = paste0("[", round(ci_low, 2), ", ", round(ci_high, 2), "]"),
           icc = round(icc, 2)) %>%
    select(rater, rubric, grade_type, ICC = icc, `95% CI` = CI),
  read_csv(paste0(out, "consistency.csv")) %>%
    mutate(across(c(mean_sd, sd_sd), ~round(.x, 2))) %>%
    select(rater, rubric, grade_type, `Mean SD` = mean_sd, `SD of SD` = sd_sd),
  by = c("rater", "rubric", "grade_type")
) %>%
  mutate(
    rater = factor(rater, levels = c("gpt-5.6-luna", "gpt-5-nano"),
                   labels = c("Luna", "Nano")),
    rubric = str_to_title(rubric),
    grade_type = factor(grade_type,
                        levels = c("final_grade", section_cols),
                        labels = c("Final grade", section_labels))
  ) %>%
  select(Model = rater, Rubric = rubric, `Grade type` = grade_type,
         ICC, `95% CI`, `Mean SD`, `SD of SD`) %>%
  arrange(Model, Rubric, `Grade type`) %>%
  flextable() %>%
  merge_v(j = c("Model", "Rubric")) %>%
  apa_table(note = "ICC(2,k) absolute agreement estimated across three runs per 
            pipeline. Mean SD = average within-student standard deviation across 
            runs. Benchmarks: < .50 poor, .50-.75 moderate, .75-.90 good, > .90 
            excellent (Koo & Li, 2016).") %>%
  save_as_docx(path = paste0(out, "table2_consistency.docx"))

# Table 3: Bias tests
read_csv(paste0(out, "bias_ttests.csv")) %>%
  mutate(
    CI = paste0("[", round(ci_low, 2), ", ", round(ci_high, 2), "]"),
    across(c(mean_diff, sd_diff, delta_median, BF10), ~round(.x, 2))
  ) %>%
  select(Comparison = comparison, `Mean diff` = mean_diff, `SD diff` = sd_diff,
         `delta (median)` = delta_median, `95% CI` = CI, BF10) %>%
  flextable() %>%
  apa_table(note = "diff = LLM minus human grade. delta = standardized effect size 
            from posterior. BF10 = Bayes factor in favor of H1 
            (true mean difference != 0).") %>%
  save_as_docx(path = paste0(out, "table3_bias_tests.docx"))

# Table 4: Proportional bias
read_csv(paste0(out, "proportional_bias.csv")) %>%
  mutate(CI = paste0("[", round(ci_low, 2), ", ", round(ci_high, 2), "]"),
         across(c(slope_median, p_neg), ~round(.x, 2))) %>%
  select(Comparison = comparison, `b (median)` = slope_median,
         `95% CI` = CI, `P(b < 0)` = p_neg) %>%
  flextable() %>%
  apa_table(note = "b = posterior median slope of Bland-Altman difference on pair 
            mean. Negative slopes indicate underscoring of high-performing 
            students and overscoring of low-performing ones. P(b < 0) = posterior 
            probability that the slope is negative.") %>%
  save_as_docx(path = paste0(out, "table4_proportional_bias.docx"))

# Table 5: Correlations
read_csv(paste0(out, "correlations.csv")) %>%
  mutate(CI = paste0("[", round(ci_low, 2), ", ", round(ci_high, 2), "]"),
         across(c(rho_median, BF10), ~round(.x, 2))) %>%
  select(Comparison = comparison, `rho (median)` = rho_median,
         `95% CI` = CI, BF10 = BF10) %>%
  flextable() %>%
  apa_table(note = "BF10 = Bayes factor in favor of H1 (presence of true correlation).") %>%
  save_as_docx(path = paste0(out, "table5_correlations.docx"))

# Table 6: Fisher z correlation comparisons
read_csv(paste0(out, "fisher_z_comparisons.csv")) %>%
  mutate(across(where(is.numeric), ~round(.x, 3))) %>%
  rename(Comparison = comparison,
         `r (a)` = r_a, `r (b)` = r_b,
         `z diff` = z_diff,
         `95% CI low` = ci_low, `95% CI high` = ci_high,
         p = p) %>%
  flextable() %>%
  apa_table(note = "z diff = Fisher z(r_a) - Fisher z(r_b). Positive values 
            indicate r_a > r_b. p = two-tailed test of H0: rho_a = rho_b.") %>%
  save_as_docx(path = paste0(out, "table6_fisher_z.docx"))

# Table 7: Agreement model summary
read_csv(paste0(out, "agreement_model_summary.csv")) %>%
  mutate(
    CI = paste0("[", round(ci_low, 2), ", ", round(ci_high, 2), "]"),
    across(c(median, p_pos, p_neg), ~round(.x, 2))
  ) %>%
  select(Criterion = human_criterion, Parameter = parameter,
         `b (median)` = median, `95% CI` = CI,
         `P(b > 0)` = p_pos, `P(b < 0)` = p_neg) %>%
  flextable() %>%
  merge_v(j = "Criterion") %>%
  apa_table(note = "Reference levels: nano, original rubric. Interactions with 
            human grade reflect moderation of slope (rank-order agreement).") %>%
  save_as_docx(path = paste0(out, "table7_agreement_model.docx"))

# Table 8: Variance components (reshaped)
read_csv(paste0(out, "variance_components.csv")) %>%
  pivot_longer(-model,
               names_to = c("quantity", ".value"),
               names_pattern = "(.+)_(median|ci_low|ci_high)") %>%
  mutate(
    CI = paste0("[", round(ci_low, 3), ", ", round(ci_high, 3), "]"),
    median = round(median, 3),
    quantity = recode(quantity,
                      "sd_student" = "SD (student)",
                      "sigma" = "SD (residual)",
                      "var_student" = "Var (student)",
                      "var_resid" = "Var (residual)",
                      "icc_student" = "ICC (student)")
  ) %>%
  select(Model = model, Quantity = quantity,
         Median = median, `95% CI` = CI) %>%
  flextable() %>%
  merge_v(j = "Model") %>%
  apa_table(note = "ICC (student) = proportion of total variance attributable 
            to student ability. Models: 1a = all raters; 1b = LLM original 
            rubric only; 1c = LLM improved rubric only; 1d = LLM both rubrics.") %>%
  save_as_docx(path = paste0(out, "table8_variance_components.docx"))

# Table 9: Variance fixed effects
read_csv(paste0(out, "variance_fixed_effects.csv")) %>%
  mutate(
    CI = paste0("[", round(ci_low, 2), ", ", round(ci_high, 2), "]"),
    across(c(median, p_pos, p_neg), ~round(.x, 2))
  ) %>%
  select(Model = model, Parameter = parameter,
         `b (median)` = median, `95% CI` = CI,
         `P(b > 0)` = p_pos, `P(b < 0)` = p_neg) %>%
  flextable() %>%
  merge_v(j = "Model") %>%
  apa_table() %>%
  save_as_docx(path = paste0(out, "table9_variance_fixed_effects.docx"))

# Table 10: Section mean differences
read_csv(paste0(out, "section_differences.csv")) %>%
  mutate(rater = factor(rater, levels = c("gpt-5.6-luna", "gpt-5-nano"),
                        labels = c("Luna", "Nano")),
         rubric = str_to_title(rubric),
         mean_diff = round(mean_diff, 2)) %>%
  pivot_wider(names_from = c(rater, rubric), values_from = mean_diff) %>%
  rename(Section = section) %>%
  flextable() %>%
  apa_table(note = "Values are LLM minus regrader mean grade. Negative values 
            indicate LLM underscoring.") %>%
  save_as_docx(path = paste0(out, "table10_section_differences.docx"))

# Table 11: Section SD ratios
read_csv(paste0(out, "section_sd_ratios.csv")) %>%
  mutate(rater = factor(rater, levels = c("gpt-5.6-luna", "gpt-5-nano"),
                        labels = c("Luna", "Nano")),
         rubric = str_to_title(rubric),
         sd_ratio = round(sd_ratio, 2)) %>%
  select(section, rater, rubric, sd_ratio) %>%
  pivot_wider(names_from = c(rater, rubric), values_from = sd_ratio) %>%
  rename(Section = section) %>%
  flextable() %>%
  apa_table(note = "Values > 1 indicate LLM grade compression relative to the regrader.") %>%
  save_as_docx(path = paste0(out, "table11_section_sd_ratios.docx"))

# Figure 1: Grade distributions
dist_plot <- all_grades %>%
  pivot_longer(-student_id, names_to = "rater", values_to = "grade") %>%
  drop_na() %>%
  mutate(rater = factor(rater, levels = rater_levels, labels = rater_labels)) %>%
  ggplot(aes(x = grade, fill = rater, color = rater)) +
  geom_density(alpha = 0.4) +
  geom_rug(alpha = 0.4) +
  scale_fill_viridis_d(option = "H") +
  scale_color_viridis_d(option = "H") +
  guides(color = "none") +
  labs(x = "Final grade", y = "Density", fill = "Rater") +
  theme(legend.position = "right")

ggsave(paste0(out, "figure1_distributions.png"), dist_plot,
       width = 17.8, height = 10, units = "cm", dpi = 300)

# Figure 2: Bland-Altman plots
ba_plot <- ba_data_raw %>%
  left_join(ba_summary, by = "comparison") %>%
  ggplot(aes(x = mean_pair, y = diff)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  geom_hline(aes(yintercept = mean_diff)) +
  geom_hline(aes(yintercept = loa_lower), linetype = "dotted") +
  geom_hline(aes(yintercept = loa_upper), linetype = "dotted") +
  geom_point(shape = 1, size = 2) +
  facet_wrap(~comparison, ncol = 3) +
  labs(x = "Mean of two raters", y = "Difference (y - x)") +
  theme(strip.background = element_blank())

ggsave(paste0(out, "figure2_bland_altman.png"), ba_plot,
       width = 22, height = 18, units = "cm", dpi = 300)

# Figure 3: Section differences
section_plot <- read_csv(paste0(out, "section_differences.csv")) %>%
  mutate(rater = factor(rater, levels = c("gpt-5.6-luna", "gpt-5-nano"),
                        labels = c("Luna", "Nano")),
         rubric = str_to_title(rubric),
         section = factor(section, levels = section_labels)) %>%
  ggplot(aes(x = section, y = mean_diff, color = rubric, shape = rater,
             group = interaction(rater, rubric))) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey") +
  geom_point(size = 3, position = position_dodge(width = 0.4)) +
  geom_line(position = position_dodge(width = 0.4), linewidth = 0.6) +
  scale_color_manual(values = c("Improved" = "blue", "Original" = "red")) +
  labs(x = NULL, y = "Mean difference (LLM - regrader)",
       color = "Rubric", shape = "Model") +
  theme(axis.text.x = element_text(angle = 20, hjust = 1),
        legend.position = "right")

ggsave(paste0(out, "figure3_section_differences.png"), section_plot,
       width = 17.8, height = 10, units = "cm", dpi = 300)