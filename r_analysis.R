
# Combining raw results with original grades and preliminary graph-------------


library(tidyverse)
teacher_grades <- read.csv2("allinfo_pseudo.csv")
LLM_grades <- read.csv("results.csv")

teacher_grades = teacher_grades %>% rename(student_id = PseudoID)

combined = LLM_grades %>% left_join(teacher_grades) %>%
  mutate(run = str_extract(run_id, "(?<=run)\\d+")) 

combined %>% ggplot(aes(x=Grade, y=final_grade, color = factor(student_id))) + 
 geom_point() +
  geom_abline(slope=1) +
  ylim(c(5,9)) +
  xlim(c(5,9)) +
  facet_wrap(~pipeline_id) +
  theme_bw()

write.csv(combined, "combined.csv")

combined %>%
  group_by(pipeline_id, student_id) %>%
  summarise(r=cor(Grade, final_grade),
            sd=sd(final_grade)) 
  select(Grade, final_grade)
library(dplyr)
library(readr)
library(stringr)
  
  

# Graph -------------------------------------------------------------------

  library(tidyverse)
  
# Load data
  
  df <- read_csv("combined_with_regrades.csv")
  
# Prepare human grades per student
  
  human <- df %>%
    group_by(student_id) %>%
    summarise(
      original_grade = first(Grade),        # original human final grade
      regrade_grade  = first(regrade_final),# regrade human final grade
      human_min      = pmin(original_grade, regrade_grade),
      human_max      = pmax(original_grade, regrade_grade),
      .groups = "drop"
    )
  
# Order students by original grade for clearer visualization
  human <- human %>%
    arrange(original_grade) %>%
    mutate(student_order = row_number())
  
  
# Prepare LLM grades per run
  
  llm <- df %>%
    select(student_id, pipeline_id, repetition, final_grade) %>%
    distinct() %>%
    inner_join(human %>% select(student_id, student_order), by = "student_id")
  
  
# Plot: human range (segments) + LLM grades (points)
  
  ggplot() +
    # Human grade range: vertical segment per student
    geom_segment(
      data = human,
      aes(x = student_order, xend = student_order,
          y = human_min,     yend = human_max),
      colour = "grey50",
      linewidth = 0.6
    ) +
    # LLM grades: points per pipeline & repetition
    geom_point(
      data = llm,
      aes(x = student_order, y = final_grade,
          colour = pipeline_id,
          shape  = repetition),
      position = position_jitter(width = 0.1, height = 0),
      size = 2
    ) +
    # Show student IDs on x-axis 
    scale_x_continuous(
      breaks = human$student_order,
      labels = human$student_id
    ) +
    labs(
      x = "Student ID",
      y = "Grade",
      colour = "Pipeline",
      shape  = "Repetition",
      title = "Human grade range and LLM grades per student"
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)
    )


# Short consistency metrics -----------------------------------------------------

  library(dplyr)
  library(readr)
  
  df <- read_csv("combined_with_regrades.csv")
  
  # One row per student × pipeline with range of final grades across runs
  ranges <- df %>%
    group_by(student_id, pipeline_id) %>%
    summarise(
      min_grade = min(final_grade, na.rm = TRUE),
      max_grade = max(final_grade, na.rm = TRUE),
      range     = max_grade - min_grade,
      .groups   = "drop"
    )
  
  # Summary per pipeline (original vs improved)
  ranges %>%
    group_by(pipeline_id) %>%
    summarise(
      n_students = n(),
      min_range  = min(range),
      mean_range = mean(range),
      max_range  = max(range),
      sd_range   = sd(range)
    )

# Data prepping for more detailed consistency metrics -------------------------

  library(tidyverse)
  
  df <- read_csv("combined_with_regrades.csv") %>%
    filter(model == "gpt-5-nano")  # explicit, even if it’s already all nano
  
  # Section + final grade columns from the LLM
  section_cols <- c(
    "intro_grade",
    "methods_grade",
    "results_grade",
    "discussion_grade",
    "lang_style_grade",
    "final_grade"
  )
  
  # Long format: one row per student x pipeline x repetition x section
  llm_long <- df %>%
    select(student_id, pipeline_id, repetition, all_of(section_cols)) %>%
    pivot_longer(
      cols = all_of(section_cols),
      names_to = "section",
      values_to = "grade"
    ) %>%
    mutate(
      section = recode(section,
                       intro_grade       = "Intro",
                       methods_grade     = "Methods",
                       results_grade     = "Results",
                       discussion_grade  = "Discussion",
                       lang_style_grade  = "Language/style",
                       final_grade       = "Final"
      )
    )
  

# Per–student x pipeline x section consistency ----------------------------
  
  # One row per student x pipeline × section:
  # variation across the three repetitions
  consistency_per_student <- llm_long %>%
    group_by(student_id, pipeline_id, section) %>%
    summarise(
      n_reps   = n(),
      min_grade = min(grade, na.rm = TRUE),
      max_grade = max(grade, na.rm = TRUE),
      range     = max_grade - min_grade,
      sd_grade  = sd(grade, na.rm = TRUE),
      .groups   = "drop"
    )
  
  # Quick look
  consistency_per_student %>% head()
  

# Per pipeline x section summary -----------------------------------------------

  consistency_summary <- consistency_per_student %>%
    group_by(pipeline_id, section) %>%
    summarise(
      n_students = n(),
      min_range  = min(range, na.rm = TRUE),
      q1_range   = quantile(range, 0.25, na.rm = TRUE),
      median_range = median(range, na.rm = TRUE),
      mean_range = mean(range, na.rm = TRUE),
      q3_range   = quantile(range, 0.75, na.rm = TRUE),
      max_range  = max(range, na.rm = TRUE),
      sd_range   = sd(range, na.rm = TRUE),
      .groups    = "drop"
    ) %>%
    arrange(section, pipeline_id)
  
  # consistency_summary
  

  