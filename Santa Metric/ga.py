import pandas as pd
import numpy as np
import pickle

from algorithm import genetic

df_sample = pd.read_csv("sample_submission.csv", index_col = 0)
sample = df_sample.text.to_list()[1]

optimizr = genetic(sample, initial_times = 10000, cross_size = 150, max_stack = 5, crossover_method = "mixture", parent_size = 42, mutation_chances = 2, batch_size = 512)
best_genome = optimizr.reputation(rep_times = 50)

with open("retry.pkl", "wb") as f :
    pickle.dump(best_genome, f)