import pandas as pd
import pickle

from algorithm import genetic

df_sample = pd.read_csv("sample_submission.csv", index_col = 0)
sample = df_sample.text.to_list()[5]

optimizr = genetic(sample, initial_times = 10000, cross_size = 80, cross_area = [25, 50, 100], max_stack = 3, crossover_method = "mixture", parent_size = 24, mutation_chances = 3, batch_size = 128, dupl = True)
best_genome = optimizr.reputation(rep_times = 200)

with open("sample5.pkl", "wb") as f :
    pickle.dump(best_genome, f)