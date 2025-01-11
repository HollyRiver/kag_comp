import pandas as pd
import pickle

from algorithm import genetic

df_sample = pd.read_csv("sample_submission.csv", index_col = 0)
sample = df_sample.text.to_list()[3]

optimizr = genetic(sample, initial_times = 10000, cross_size = 120, cross_area = [7, 15, 30], max_stack = 5, crossover_method = "mixture", parent_size = 42, mutation_chances = 2, batch_size = 1024, dupl = True)
best_genome = optimizr.reputation(rep_times = 200)

with open("sample3.pkl", "wb") as f :
    pickle.dump(best_genome, f)