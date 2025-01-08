import itertools
import numpy as np
from algorithm.kaggle_evaluate import PerplexityCalculator
from typing import List

class genetic :
    def __init__(self, sample : str, initial_times = 1000, max_stack = 20, cross_area = [5, 10, 20], cross_size = 1000, mutation_chances = 2, dupl = False, crossover_method = "roulette", parent_size = 10) :
        ## create evaluator
        self.evaluatr = PerplexityCalculator("google/gemma-2-9b")
        
        ## genome generate
        self.genomes = ["" for _ in range(initial_times)]
                
        for i in range(initial_times) :
            self.genomes[i] = np.random.permutation(sample.split())
        
        self.genomes = np.array(self.genomes)
        
        ## parameter setting
        self.max_stack = max_stack
        self.stack = 0
        
        self.mutation_chances = mutation_chances
        self.cross_area = cross_area
        self.cross_size = cross_size
        self.crossover_method = crossover_method
        self.dupl = dupl
        
        ## getting perplexities and parents
        perplexities = np.array(self.evaluatr.get_perplexity([" ".join(genome) for genome in self.genomes], batch_size = 1024))
        
        if crossover_method == "roulette" :
            per_sum = sum(1/(perplexities**3))
            proba = 1/(perplexities**3)/per_sum
            self.parents_indx = np.random.choice([i for i in range(initial_times)], p = proba, size = parent_size, replace = False)
            
        elif crossover_method == "rank" :
            self.ranking = parent_size
            self.parents_indx = perplexities.argsort()[:parent_size]
        
        self.best_genome = [self.genomes[perplexities.argmin()], perplexities.min()]
    
    def crossover(self, parents_indx : List[int], genomes : List[str], dupl = False) :
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
                for j in range(self.cross_size) :
                    if np.random.random() < 0.2 :
                        width = np.random.randint(self.cross_area[0], self.cross_area[2])
                        
                    else :
                        width = np.random.randint(self.cross_area[0], self.cross_area[1])
                        
                    start = np.random.randint(0, self.cross_area[2]-width)
                    child = [" " for _ in range(self.cross_area[2])]
                    child[start:start+width] = parents[i][start:start+width]  ## child에 교차영역 복사
                    
                    ## mapping
                    mapping_set = set(parents[1-i][start:start+width]) - set(parents[i][start:start+width])
                    
                    for t in mapping_set :
                        sub = np.where(np.array(parents[1-i]) == t)
                        
                        for k in range(self.cross_area[2]) :
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

    def mutation(self, childs : List[str]) -> List[str] :
        childs = np.array(childs)
        lnth = len(childs[0])
        
        for i in range(len(childs)) :
            for _ in range(self.mutation_chances) :
                ## mutate randomly
                if np.random.random() < 0.5 :
                    continue
                
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
    
    def reputation(self, rep_times = 100) :
        for _ in range(rep_times) :
            childs = self.crossover(self.parents_indx, self.genomes, self.dupl)
            self.genomes = self.mutation(childs)
            
            self.evaluatr.clear_gpu_memory()
            self.evaluatr = PerplexityCalculator("google/gemma-2-9b")
            
            perplexities = np.array(self.evaluatr.get_perplexity([" ".join(genome) for genome in self.genomes], batch_size = 1024))
            
            if self.crossover_method == "roulette" :
                per_sum = sum(1/(perplexities**3))
                proba = 1/(perplexities**3)/per_sum
                self.parents_indx = np.random.choice([i for i in range(len(self.genomes))], p = proba, size = 10, replace = False)
                
            elif self.crossover_method == "rank" :
                self.parents_indx = perplexities.argsort()[:self.ranking]
            
            if perplexities.min() < self.best_genome[1] :
                self.best_genome = [self.genomes[perplexities.argmin()], perplexities.min()]
                print(f"best genome : {self.best_genome}")
                self.stack = 0
                
            else :
                self.stack += 1
                
                if self.stack >= self.max_stack :
                    break
                
            print(f"parents perplexities : {perplexities[self.parents_indx]}")
            
        return self.best_genome