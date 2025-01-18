import pandas as pd
import pickle

from algorithm import genetic

df_sample = pd.read_csv("sample_submission.csv", index_col = 0)
sample2 = pd.read_csv("sample_submission.csv", index_col = 0).text.to_list()[2]

optimizr = genetic(sample2, initial_times = 10000, max_stack = 10, cross_size = 200, crossover_method = "mixture", elite_size = 5, parent_size = 50, batch_size = 512)
best_genome = optimizr.reputation(rep_times = 200)

with open("sample2.pkl", "wb") as f :
    pickle.dump(best_genome, f)