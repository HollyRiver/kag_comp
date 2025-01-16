import numpy as np
from algorithm.kaggle_evaluate import PerplexityCalculator
from typing import List

class genetic_neo :
    def __init__(
        self,
        sample : str,
        verbs : List[str],
        initial_times = 1000,
        max_stack = 10,
        cross_size = 10,
        mutation_chances = 1,
        dupl = False,
        crossover_method = "mixture",
        parents_size = 10,
        elite_size = 5,
        batch_size = 1024,
        load_in_8bit = False) :
        
        
        ## create evaluator
        self.batch_size = batch_size
        self.load_in_8bit = load_in_8bit
        self.evaluatr = PerplexityCalculator("google/gemma-2-9b", load_in_8bit = self.load_in_8bit)
        
        ## first genomes generate
        self.genomes = ["" for _ in range(initial_times)]
                
        for i in range(initial_times) :
            self.genomes[i] = np.random.permutation(sample.split())
        
        self.genomes = np.array(self.genomes) ## numpy array로 변경
        
        ## parameter setting
        self.max_stack = max_stack
        self.stack = 0  ## current stack
        
        self.mutation_chances = mutation_chances    ## mutation chance in crossover
        self.cross_size = cross_size    ## crossover sample size
        self.parents_size = parents_size
        self.crossover_method = crossover_method
        self.vrbs = verbs
        self.dupl = dupl
        
        ## use only mixture method
        if self.crossover_method == "mixture" :
            self.elite_size = elite_size
        
        ## getting perplexities and parents
        perplexities = np.array(self.evaluatr.get_perplexity([" ".join(genome) for genome in self.genomes], batch_size = self.batch_size))
        
        ## abstract parents
        self.parents_indx = self.selection(perplexities, parents_size = self.parents_size, crossover_method = self.crossover_method, elite_size = self.elite_size)  ## abstraction parents index
        self.best_genome = [" ".join(self.genomes[perplexities.argmin()]), perplexities.min()]  ## receive best_genome
        
        print(f"parents perplexities : {perplexities[self.parents_indx]}")
        
        
    def selection(self, perplexities, crossover_method = "mixture", parents_size = 20, elite_size = 10) :
        perps = np.array(perplexities)
        
        if crossover_method == "roulette" :
            per_sum = sum(1/(perps**2))
            proba = 1/(perps**2)/per_sum
            parents_indx = np.random.choice([i for i in range(len(perps))], p = proba, size = parents_size, replace = False)
            
        elif crossover_method == "rank" :
            parents_indx = perps.argsort()[:parents_size]

        elif crossover_method == "mixture" :
            per_sum = sum(1/(perps**2))
            proba = 1/(perps**2)/per_sum
            subset_index = np.random.choice([i for i in range(len(perps))], p = proba, size = parents_size - elite_size, replace = False)
            sorted_index = perps.argsort()
            remain_size = elite_size
            
            for i in range(parents_size) :
                if sorted_index[i] not in subset_index :
                    subset_index = np.concat([subset_index, [sorted_index[i]]])
                    remain_size -= 1
                    
                    if remain_size == 0 :
                        break
                
            parents_indx = subset_index
        
        return parents_indx
    
    
    def mutation_crossover(self, p) :
        origin = np.array(p)
        lnth = len(origin)
        
        for _ in range(self.mutation_chances) :
            ## mutation rate : default 50%
            if np.random.random() > 0.33 :
                dice = np.random.randint(0, 4) ## choice mutation method
                
                ## swap
                if dice == 0 :
                    swap_area = np.random.choice([k for k in range(lnth)], size = 2, replace = False)
                    origin[swap_area[0]], origin[swap_area[1]] = origin[swap_area[1]], origin[swap_area[0]]
                    
                ## move
                elif dice == 1 :
                    moving_indx = np.random.randint(0, lnth)
                    mover = origin[moving_indx]
                    
                    moving_area = np.random.randint(0, lnth)
                    
                    ## trick
                    tmp = list(origin)
                    del tmp[moving_indx]
                    tmp.insert(moving_area, mover)
                    
                    origin = np.array(tmp)
                    
                ## inverse
                elif dice == 2 :
                    if np.random.random() < 0.5 :
                        width = np.random.randint(3, 5)
                        start = np.random.randint(0, lnth-width)
                        origin[start:start+width] = origin[start:start+width][::-1]
                    
                ## scramble
                elif dice == 3 :
                    if np.random.random() < 0.5 :
                        swap_size = np.random.randint(3, 5)
                        swap_area = np.random.choice([i for i in range(lnth)], size = swap_size, replace = False)
                        
                        origin[swap_area] = np.random.permutation(np.array(origin)[swap_area])
        
        return origin
    
    
    def crossover(self, p1, p2, p3, verbs, mutation_chances = 1) :
        structure = [None for _ in range(len(p1))]
        vrbs = [t for t in p2 if t in verbs]
        othrs = [t for t in p3 if t not in verbs]
        
        child = ["" for _ in range(len(p1))]
        
        for i, t in enumerate(p1) :
            structure[i] = t in verbs
        
        if mutation_chances > 0 :
            structure = self.mutation_crossover(structure)
            vrbs = self.mutation_crossover(vrbs)
            othrs = self.mutation_crossover(othrs)

        a = 0
        b = 0
        
        for i, s in enumerate(structure) :
            if s :
                child[i] = vrbs[a]
                a += 1
            else :
                child[i] = othrs[b]
                b += 1
                
        return child
    

    def mutation(self, genome, mutation_chances = 2) :
        origin = np.array(genome)
        lnth = len(origin)
        
        for _ in range(mutation_chances) :
            ## mutate randomly
            if np.random.random() > 0.5 :
                dice = np.random.randint(0, 4)
                
                ## swap
                if dice == 0 :
                    swap_area = np.random.choice([k for k in range(lnth)], size = 2, replace = False)
                    origin[swap_area[0]], origin[swap_area[1]] = origin[swap_area[1]], origin[swap_area[0]]
                
                ## move
                elif dice == 1 :
                    moving_indx = np.random.randint(0, lnth)
                    mover = origin[moving_indx]
                    
                    moving_area = np.random.randint(0, lnth)
                    
                    ## trick
                    tmp = list(origin)
                    del tmp[moving_indx]
                    tmp.insert(moving_area, mover)
                    
                    origin = np.array(tmp)
                    
                ## inverse
                elif dice == 2 :
                    if np.random.random() < 0.5 :
                        width = np.random.randint(3, 5)
                        start = np.random.randint(0, lnth-width)
                        origin[start:start+width] = origin[start:start+width][::-1]
                    
                ## scramble
                elif dice == 3 :
                    if np.random.random() < 0.5 :
                        swap_size = np.random.randint(3, 5)
                        swap_area = np.random.choice([i for i in range(lnth)], size = swap_size, replace = False)
                        
                        origin[swap_area] = np.random.permutation(np.array(origin)[swap_area])
        
        return origin
    
    
    def reputation(self, rep_times = 100) :
        ## initialize
        stack = 0
        genome_set = self.genomes   ## initial genomes : numpy.array
        best_genome = self.best_genome
        parents_indx = self.parents_indx
        
        for i in range(rep_times) :
            pair_parents = []
            
            ## generate parents sets
            for p1 in parents_indx :
                for p2 in parents_indx :
                    for p3 in parents_indx :
                        pair_parents.append([p1, p2, p3])
            
            childs = []
            
            ## crossover & mutation
            for pair in pair_parents :
                parents_genome = [genome_set[idx] for idx in pair]
                
                for _ in range(self.cross_size) :
                    if stack <= self.max_stack//4 :
                        crossover_genome = self.crossover(parents_genome[0], parents_genome[1], parents_genome[2], verbs = self.vrbs, mutation_chances = self.mutation_chances)
                        childs.append(self.mutation(crossover_genome, mutation_chances = 4))
                        
                    else : 
                        crossover_genome = self.crossover(parents_genome[0], parents_genome[1], parents_genome[2], verbs = self.vrbs, mutation_chances = self.mutation_chances*2)
                        childs.append(self.mutation(crossover_genome, mutation_chances = 8))
                                               
            
            ## setting new genome set
            genome_set = np.unique([" ".join(genome) for genome in childs] + [best_genome[0]]) ## saving best genome in genome set

            ## evaluate
            self.evaluatr.clear_gpu_memory()
            self.evaluatr = PerplexityCalculator("google/gemma-2-9b")
            
            perplexities = np.array(self.evaluatr.get_perplexity(genome_set, batch_size = self.batch_size))
            
            ## select parents
            parents_indx = self.selection(perplexities, parents_size = self.parents_size, crossover_method = self.crossover_method, elite_size = self.elite_size)
            
            ## renewal
            if perplexities.min() < best_genome[1] :
                best_genome = [genome_set[perplexities.argmin()], perplexities.min()]
                print(f"best genome : {best_genome}")
                stack = 0
                
            else :
                stack += 1
                
                if stack >= self.max_stack :
                    break
                
            ## genome set formatting
            genome_set = [genome.split() for genome in genome_set]
            
            print(f"parents perplexities : {perplexities[parents_indx]}")
            
        return best_genome