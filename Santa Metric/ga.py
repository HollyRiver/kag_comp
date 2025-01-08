import pandas as pd
import numpy as np
import pickle

from algorithm import PerplexityCalculator
from algorithm import genetic

df_sample = pd.read_csv("sample_submission.csv", index_col = 0)
sample = df_sample.text.to_list()[1]

optimizr = genetic(sample, cross_size = 300, crossover_method = "rank")
best_genome = optimizr.reputation(rep_times = 500)

with open("ga.pkl", "wb") as f :
    pickle.dump(best_genome, f)