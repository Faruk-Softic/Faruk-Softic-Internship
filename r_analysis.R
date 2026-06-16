library(tidyverse)
teacher_grades <- read.csv2("allinfo_pseudo.csv")
LLM_grades <- read.csv("results.csv")

teacher_grades = teacher_grades %>% rename(student_id = PseudoID)

combined = LLM_grades %>% left_join(teacher_grades)

combined %>% ggplot(aes(x=Grade, y=final_grade, color = factor(student_id))) + 
 geom_point() +
  geom_abline(slope=1) +
  ylim(c(4,10)) +
  xlim(c(4,10)) +
  facet_wrap(~rubric_version)

write.csv(combined, "combined.csv")


library(dplyr)
library(readr)
library(stringr)

CSV_PATH <- "/path/to/your/results.csv"

new_results <- read_csv("results.csv")

new_results <- new_results |>
  mutate(run_number = as.integer(str_extract(run_id, "(?<=\\.run)\\d+")))

write_csv(new_results, "new_results.csv")

