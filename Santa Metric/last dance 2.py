import pandas as pd
import numpy as np
import itertools

from algorithm import PerplexityCalculator

sample = ['the', 'grinch', 'eat', 'not', 'you', 'scrooge', 'and', 'hohoho', 'yuletide',
  'greeting', 'have', 'merry', 'unwrap', 'give', 'as', 'we', 'the', 'of', 'to',
  'from', 'toy', 'doll', 'ornament', 'snowglobe', 'card', 'game', 'puzzle',
  'advent', 'candle', 'wreath', 'cookie', 'bake', 'fruitcake', 'gingerbread',
  'candy', 'peppermint', 'chocolate', 'milk', 'and', 'eggnog', 'sleigh', 'drive',
  'reindeer', 'jump', 'polar', 'kaggle', 'beard', 'elf', 'workshop', 'workshop',
  'naughty', 'nice', 'wrapping', 'paper', 'bow', 'ornament', 'nutcracker',
  'poinsettia', 'holly', 'mistletoe', 'jingle', 'relax', 'family', 'laugh',
  'joy', 'peace', 'sleep', 'dream', 'wish', 'hope', 'believe', 'wonder', 'magi',
  'visit', 'with', 'gifts', 'in', 'stocking', 'chimney', 'fireplace',
  'fireplace', 'chimney', 'star', 'angel', 'night', 'night', 'walk', 'carol',
  'sing', 'holiday', 'decorations', 'that', 'it', 'is', 'the', 'season', 'of',
  'cheer', 'and', 'cheer']

evaluatr = PerplexityCalculator("google/gemma-2-9b")
best_perplexity = evaluatr.get_perplexity(" ".join(sample), batch_size = 4)
evaluatr.clear_gpu_memory()

## all neighborhoods
neighborhoods = []

for set in itertools.combinations([i for i in range(100)], 3) :
    for sub_set in itertools.permutations(set) :
        neighborhood = sample.copy()
        
        for i, idx in enumerate(sorted(sub_set)) :
            neighborhood[idx] = sample[sub_set[i]]
            
        neighborhoods.append(neighborhood)
neighborhoods = np.unique([" ".join(neighborhood) for neighborhood in neighborhoods])

print(len(neighborhoods))

evaluatr = PerplexityCalculator("google/gemma-2-9b")
perplexities = np.array(evaluatr.get_perplexity(neighborhoods, batch_size = 128))
    
if perplexities.min() < best_perplexity :
    print(f"new best : {[neighborhoods[perplexities.argmin()], perplexities.min()]}")

else :
    print("was not able to find new best set in neighbor size 3!")