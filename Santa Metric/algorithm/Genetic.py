import itertools
import numpy as np
from algorithm.kaggle_evaluate import PerplexityCalculator
from typing import List

class genetic :
    def __init__(self, sample : str, initial_times = 1000, max_stack = 20, cross_area = [5, 10, 20], cross_size = 1000, mutation_chances = 2, dupl = False, crossover_method = "roulette", elite_size = None, parent_size = 10, batch_size = 1024, load_in_8bit = False) :
        ## create evaluator
        self.batch_size = batch_size
        self.load_in_8bit = load_in_8bit
        self.evaluatr = PerplexityCalculator("google/gemma-2-9b", load_in_8bit = self.load_in_8bit)
        
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
        perplexities = np.array(self.evaluatr.get_perplexity([" ".join(genome) for genome in self.genomes], batch_size = self.batch_size))
        
        if crossover_method == "roulette" :
            per_sum = sum(1/(perplexities**2))
            proba = 1/(perplexities**2)/per_sum
            self.parents_indx = np.random.choice([i for i in range(initial_times)], p = proba, size = parent_size, replace = False)
            
        elif crossover_method == "rank" :
            self.ranking = parent_size
            self.parents_indx = perplexities.argsort()[:parent_size]
            
        elif crossover_method == "mixture" :
            self.ranking = parent_size
            
            if elite_size == None :
                self.subset_size = self.ranking//2
                self.remain_size = self.ranking - self.subset_size
            
            else :
                self.subset_size = self.ranking - elite_size
                self.remain_size = elite_size
            
            per_sum = sum(1/(perplexities**2))
            proba = 1/(perplexities**2)/per_sum
            
            subset_index = np.random.choice([i for i in range(initial_times)], p = proba, size = self.subset_size, replace = False)
            sorted_index = perplexities.argsort()
            
            remain_size = self.remain_size
            
            for i in range(self.ranking) :
                if sorted_index[i] not in subset_index :
                    subset_index = np.concat([subset_index, [sorted_index[i]]])
                    remain_size -= 1
                    
                    if remain_size == 0 :
                        break
                
            self.parents_indx = subset_index
        
        self.best_genome = [" ".join(self.genomes[perplexities.argmin()]), perplexities.min()]
        
        print(f"parents perplexities : {perplexities[self.parents_indx]}")
    
    def crossover(self, parents_indx : List[int], genomes : List[str], dupl = False) :
        pair_parents = list(itertools.combinations(parents_indx, 2)) ## combination of parents    
        childs = []
        
        for pair in pair_parents :
            parents = [genomes[pair[0]], genomes[pair[1]]]
            
            ## dealing with duplication
            if dupl :
                for i in range(2) :
                    for t in set(parents[i]) :
                        times = sum(np.array(parents[i]) == t)
                        rep = 1
                        
                        if times > 1 :
                            for j, k in enumerate(parents[i]) :
                                if k == t :
                                    parents[i][j] = f"{t}{rep}"
                                    rep += 1
            
            # PMX : i is main / 1-i is sub
            for i in range(2) :
                for j in range(self.cross_size) :
                    if np.random.random() < 0.4 :
                        width = np.random.randint(self.cross_area[0], self.cross_area[2])
                        
                    else :
                        width = np.random.randint(self.cross_area[0], self.cross_area[1])
                        
                    start = np.random.randint(0, self.cross_area[2]-width)
                    child = [" " for _ in range(self.cross_area[2])]
                    child[start:start+width] = parents[i][start:start+width]  ## child에 교차영역 복사
                    
                    ## mapping
                    mapping_set = set(parents[1-i][start:start+width]) - set(parents[i][start:start+width])
                    
                    for t in mapping_set :
                        sub = np.where(np.array(parents[1-i]) == t)[0][0]
                        
                        for k in range(self.cross_area[2]) :
                            sub = np.where(np.array(parents[1-i]) == parents[i][sub])[0][0]
                            
                            if (sub < start) | (sub >= start+width) :
                                child[sub] = t
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
                                child[j] = t[:-1]
                            except :
                                pass
                    
                    childs.append(child)
        
        return childs

    def mutation(self, childs : List[str], mutation_chances = 2) -> List[str] :
        childs = np.array(childs)
        lnth = len(childs[0])
        
        for i in range(len(childs)) :
            for _ in range(mutation_chances) :
                ## mutate randomly
                if np.random.random() > 0.7 :
                    dice = np.random.randint(0, 4) ## roll a dice
                    
                    ## swap
                    if dice == 0 :                
                        swap_area = np.random.choice([k for k in range(lnth)], size = 2, replace = False)
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
                        if np.random.random() < 0.5 :
                            width = np.random.randint(3, 6)
                            start = np.random.randint(0, lnth-width)
                            childs[i][start:start+width] = childs[i][start:start+width][::-1]
                        
                    ## scramble
                    elif dice == 3 :
                        if np.random.random() < 0.5 :
                            swap_size = np.random.randint(3, 6)
                            swap_area = np.random.choice([i for i in range(lnth)], size = swap_size, replace = False)
                            
                            childs[i][swap_area] = np.random.permutation(np.array(childs[i])[swap_area])
            
        return childs
    
    def reputation(self, rep_times = 100) :
        parents_indx = self.parents_indx
        genomes = self.genomes
        best_genome = self.best_genome
        
        for _ in range(rep_times) :
            childs = self.crossover(parents_indx, genomes, self.dupl)
            genomes = self.mutation(childs, mutation_chances = self.mutation_chances)
            
            genome_set = np.unique([" ".join(genome) for genome in genomes] + [best_genome[0]]) ## elitism
            
            self.evaluatr.clear_gpu_memory()
            self.evaluatr = PerplexityCalculator("google/gemma-2-9b", load_in_8bit = self.load_in_8bit)
            
            perplexities = np.array(self.evaluatr.get_perplexity(genome_set, batch_size = self.batch_size))
            
            if self.crossover_method == "roulette" :
                per_sum = sum(1/(perplexities**2))
                proba = 1/(perplexities**2)/per_sum
                parents_indx = np.random.choice([i for i in range(len(genome_set))], p = proba, size = 10, replace = False)
                
            elif self.crossover_method == "rank" :
                parents_indx = perplexities.argsort()[:self.ranking]
            
            elif self.crossover_method == "mixture" :
                per_sum = sum(1/(perplexities**2))
                proba = 1/(perplexities**2)/per_sum
                subset_index = np.random.choice([i for i in range(len(genome_set))], p = proba, size = self.subset_size, replace = False)
                sorted_index = perplexities.argsort()
                
                remain_size = self.remain_size
                
                for i in range(self.ranking) :
                    if sorted_index[i] not in subset_index :
                        subset_index = np.concat([subset_index, [sorted_index[i]]])
                        remain_size -= 1
                        
                        if remain_size == 0 :
                            break
                    
                parents_indx = subset_index
            
            if perplexities.min() < best_genome[1] :
                best_genome = [genome_set[perplexities.argmin()], perplexities.min()]
                print(f"best genome : {best_genome}")
                self.stack = 0
                
            else :
                self.stack += 1
                
                if self.stack >= self.max_stack :
                    break
                
            genomes = [genome.split() for genome in genome_set]
                
            print(f"parents perplexities : {perplexities[parents_indx]}")
            
        return best_genome