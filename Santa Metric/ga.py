import pandas as pd
import numpy as np
import pickle

from algorithm import genetic

df_sample = pd.read_csv("sample_submission.csv", index_col = 0)
sample = df_sample.text.to_list()[4]

optimizr = genetic(sample, initial_times = 10000, cross_size = 100, cross_area = [12, 25, 50], max_stack = 5, crossover_method = "mixture", parent_size = 32, mutation_chances = 3, batch_size = 256, dupl = False)
best_genome = optimizr.reputation(rep_times = 200)

with open("sample4.pkl", "wb") as f :
    pickle.dump(best_genome, f)