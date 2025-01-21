import pandas as pd
import pickle

from algorithm import genetic_neo_pmx

sample = pd.read_csv("sample_submission.csv").text.to_list()[5]
verbs = ["walk", "give", "jump", "drive", "bake", "sleep", "laugh", "sing", "eat", "visit", "relax", "unwrap", "believe", "dream", "hope", "wish", "wrap", "decorate", "play", "wonder", "is", "have"]

optimizr = genetic_neo_pmx(sample, verbs, initial_times = 10000, max_stack = 8, cross_size = 16, mutation_chances = 1, dupl = True, crossover_method = "mixture", parents_size = 16, elite_size = 4, batch_size = 128)
best_genome = optimizr.reputation(rep_times = 100)

with open("sample5_neo.pkl", "wb") as f :
    pickle.dump(best_genome, f)