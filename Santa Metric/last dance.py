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

## width = 5
for k in range(50) :
    width = 5
    all_subset = []

    for i in range(len(sample) - width) :
        all_subset = all_subset + [" ".join(sample[:i] + list(set) + sample[width+i:]) for set in itertools.permutations(sample[i:width+i])]
        

    all_subset = np.unique(all_subset)
    print(len(all_subset))

    evaluatr = PerplexityCalculator("google/gemma-2-9b")
    perplexities = np.array(evaluatr.get_perplexity(all_subset, batch_size = 128))
    evaluatr.clear_gpu_memory()
        
    if perplexities.min() < best_perplexity :
        print(f"new best : {[all_subset[perplexities.argmin()], perplexities.min()]}")
        best_perplexity = perplexities.min()
        sample = all_subset[perplexities.argmin()].split()
    
    else :
        print(f"cannot find new best set in width {width}!")
        break
    


## width = 6    
width = 6
all_subset = []

for i in range(len(sample) - width) :
    all_subset = all_subset + [" ".join(sample[:i] + list(set) + sample[width+i:]) for set in itertools.permutations(sample[i:width+i])]
    

all_subset = np.unique(all_subset)
print(len(all_subset))

evaluatr = PerplexityCalculator("google/gemma-2-9b")
perplexities = np.array(evaluatr.get_perplexity(all_subset, batch_size = 128))
evaluatr.clear_gpu_memory()
    
if perplexities.min() < best_perplexity :
    print(f"new best : {[all_subset[perplexities.argmin()], perplexities.min()]}")
    best_perplexity = perplexities.min()
    sample = all_subset[perplexities.argmin()].split()

else :
    print(f"cannot find new best set in width {width}!")