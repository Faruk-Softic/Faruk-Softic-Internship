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


