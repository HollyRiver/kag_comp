import gc ## for memory
import os
import copy
import pickle
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import itertools
import pandas as pd
import transformers
import torch

from kaggle_evaluate import PerplexityCalculator

def crossover(parents_indx : List[int], genomes : List[str], cross_area : List[int], sample_num : int, dupl = False) :
    pair_parents = list(itertools.combinations(parents_indx, 2)) ## combination of parents    
    childs = []
    
    for pair in pair_parents :
        parents = [genomes[pair[0]], genomes[pair[1]]]
        
        ## dealing with duplication
        if dupl :
            for i in range(2) :
                for t in set(parents[i].split()) :
                    times = sum(np.array(parents[i].split()) == t)
                    rep = 1
                    
                    if times > 1 :
                        for j, k in enumerate(parents[i].split()) :
                            if k == t :
                                parents[i][j] = f"{t}{rep}"
                                rep += 1
        
        # PMX : i is main / 1-i is sub
        for i in range(2) :
            for j in range(1000) :
                if np.random.random() < 0.2 :
                    width = np.random.randint(cross_area[sample_num][0], cross_area[sample_num][2])
                    
                else :
                    width = np.random.randint(cross_area[sample_num][0], cross_area[sample_num][1])
                    
                start = np.random.randint(0, cross_area[sample_num][2]-width)
                child = [" " for _ in range(cross_area[sample_num][2])]
                child[start:start+width] = parents[i][start:start+width]  ## child에 교차영역 복사
                
                ## mapping
                mapping_set = set(parents[1-i][start:start+width]) - set(parents[i][start:start+width])
                
                for t in mapping_set :
                    sub = np.where(np.array(parents[1-i]) == t)
                    
                    for k in range(cross_area[sample_num][2]) :
                        sub = np.where(np.array(parents[1-i]) == parents[i][sub])
                        
                        if (sub[0] < start) | (sub[0] > start+width) :
                            child[int(sub[0])] = t
                            break
                
                ## remain set
                current_set = [t for t in child if t != " "]
                remain_set = [t for t in parents[1-i] if t not in current_set]
                
                empty_indx = [i for i, t in enumerate(child) if t == " "]
                
                for k, ind in enumerate(empty_indx) :
                    child[ind] = remain_set[k]

                ## text transformation
                if dupl :
                    for j, t in enumerate(child) :
                        try :
                            int(t[-1])
                            child[i][j] = t[:-1]
                        except :
                            pass
                
                childs.append(child)
    
    return childs

def mutation(childs : List[str], mutation_area : List[int], sample_num : int) -> List[str] :
    childs = np.array(childs)
    lnth = len(childs[0])
    
    for i in range(len(childs)) :
        for _ in range(mutation_area[sample_num]) :
            dice = np.random.randint(0, 4) ## roll a dice
            
            ## swap
            if dice == 0 :                
                swap_area = np.random.choice([i for i in range(lnth)], size = 2, replace = False)
                childs[i][swap_area[0]], childs[i][swap_area[1]] = childs[i][swap_area[1]], childs[i][swap_area[0]]
                
            ## move
            elif dice == 1 :
                moving_indx = np.random.randint(0, lnth)
                mover = childs[i][moving_indx]
                
                moving_area = np.random.randint(0, lnth-1)
                
                ## trick
                tmp = list(childs[i])
                del tmp[moving_indx]
                tmp.insert(moving_area, mover)
                
                childs[i] = np.array(tmp)
                
            ## inverse
            elif dice == 2 :
                width = np.random.randint(3, 6)
                start = np.random.randint(0, lnth-width)
                childs[i][start:start+width] = childs[i][start:start+width][::-1]
                
            ## scramble
            elif dice == 3 :
                swap_size = np.random.randint(3, 6)
                swap_area = np.random.choice([i for i in range(lnth)], size = swap_size, replace = False)
                
                childs[i][swap_area] = np.random.permutation(np.array(childs[i])[swap_area])
        
    return childs


df_sample = pd.read_csv("sample_submission.csv")
samples = df_sample.text[1:6].to_list()

## initiate
initial_times = 10000
genomes = ["" for _ in range(initial_times)]

for i in range(initial_times) :
    genomes[i] = np.random.permutation(samples[0].split())
    
genomes = np.array(genomes)

## selection
evaluatr = PerplexityCalculator("google/gemma-2-9b")

perplexities = np.array(evaluatr.get_perplexity([" ".join(genome) for genome in genomes], batch_size = 1024))
per_sum = sum(1/(perplexities**3))
proba = 1/(perplexities**3)/per_sum

best_genome = [genomes[perplexities.argmin()], perplexities.min()]
parents_indx = np.random.choice([i for i in range(initial_times)], p = proba, size = 10, replace = False)

stack = 0
cross_area = [[5, 10, 20], [5, 10, 20], [7, 15, 30], [12, 25, 50], [25, 50, 100]]
mutation_times = [3, 3, 5, 7, 10]
sample_length = len(genomes[0])

for i in range(1000) :
    ## crossover
    childs = crossover(parents_indx, genomes, cross_area, sample_num = 0, dupl = False)

    ## mutation
    genomes = mutation(childs, mutation_times, sample_num = 0)

    ## evaluation / reset
    evaluatr.clear_gpu_memory()
    evaluatr = PerplexityCalculator("google/gemma-2-9b")
    
    perplexities = np.array(evaluatr.get_perplexity([" ".join(genome) for genome in genomes], batch_size = 1024))
    per_sum = sum(1/(perplexities**3))
    proba = 1/(perplexities**3)/per_sum

    if perplexities.min() < best_genome[1] :
        best_genome = [genomes[perplexities.argmin()], perplexities.min()]
        print(f"best genome : {best_genome}")
        stack = 0

    else :
        stack += 1
        
        if stack >= 50 :
            break

    parents_indx = np.random.choice([i for i in range(len(genomes))], p = proba, size = 10, replace = False)
    
    print(f"parents perplexities : {perplexities[parents_indx]}")
    

with open("temp.pkl", "wb") as f :
    pickle.dump(best_genome, f)