library(multimode)
library(tidyverse)
library(jsonlite)

human_data_path <- "./datasets/probcopa.jsonl"

human_data <- stream_in(file(human_data_path)) %>% as_tibble()

UID_significances <- human_data %>% group_by(UID) %>% summarise(silverman_test_p_value = multimode::modetest(response, method = "SI")$p.value)

write_csv(UID_significances, "./results/ProbCOPA_silverman_test_significances.csv")
