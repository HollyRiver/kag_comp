import pickle
from algorithm import genetic_brute

verbs = ["sing", "eat", "visit", "relax", "unwrap", "is"]

## seed
samples = ["sleigh of the magi yuletide cheer is unwrap gifts and eat cheer holiday decorations holly jingle relax sing carol visit workshop grinch naughty nice chimney stocking ornament nutcracker polar beard",
            "jingle yuletide carol sing unwrap gifts relax eat holiday cheer cheer decorations ornament holly stocking naughty and nice chimney sleigh workshop visit of the magi nutcracker grinch polar is beard",
            "ornament yuletide is the unwrap of holiday decorations gifts eat relax cheer cheer carol sing holly jingle sleigh workshop naughty and nice chimney stocking nutcracker magi visit grinch polar beard"]

optimizr = genetic_brute(samples[0], verbs, initial_times = 10000, max_stack = 10, cross_size = 16, mutation_chances = 1, crossover_method = "mixture", parents_size = 4, elite_size = 3, batch_size = 512, dupl = True,
                           public_sample = samples)

best_genome = optimizr.reputation(rep_times = 15)

with open("ga_bf.pkl", "wb") as f :
    pickle.dump(best_genome, f)