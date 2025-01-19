import pandas as pd
import pickle

from algorithm import genetic_neo_pmx

samples = pd.read_csv("sample_submission.csv")
sample = samples.text.to_list()[3]

verbs = ["sing", "eat", "visit", "relax", "unwrap", "and", "of", "the", "is"] ## cheer는 제외. 불용어 추가. cheer가 두 개네...

optimizr = genetic_neo_pmx(sample, verbs, initial_times = 10000, max_stack = 5, cross_size = 8, crossover_method = "mixture", parents_size = 20, elite_size = 4, mutation_chances = 0, batch_size = 512, dupl = True)
best_genome = optimizr.reputation(rep_times = 100)

with open("new_sample3.pkl", "wb") as f :
    pickle.dump(best_genome, f)