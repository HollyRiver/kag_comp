import numpy as np
from algorithm.kaggle_evaluate import PerplexityCalculator
from typing import List

class genetic_neo_pmx :
    """_summary_
    cross_area : 
    """
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
        elite_size = 3,
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
        self.vrbs_area = [len(verbs)//(2**(2-i)) for i in range(3)]
        self.othrs_area = [(len(sample.split()) - len(verbs))//(2**(2-i)) for i in range(3)]
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
    
    
    def mutation_structure(self, p) :
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
    
    
    def dupl_cleaner(self, p, clean_type) :
        """
        for handling duplication

        Args:
            p (List[str]): token list
            clean_type (str): if input "labeling" then add numbering for duplicated token

        Returns:
            prnt (List[str])
        """
        prnt = p.copy()
        
        if clean_type == "labeling" :    
            for t in set(p) :
                times = sum(np.array(p) == t)
                rep = 1
                
                if times > 1 :
                    for j, k in enumerate(p) :
                        if k == t :
                            prnt[j] = f"{t}{rep}"
                            rep += 1
                            
        else :
            for j, t in enumerate(p) :
                try :
                    int(t[-1])
                    prnt[j] = t[:-1]
                except :
                    pass
                        
        return prnt
    
    
    def PMX(self, p1, p2, mapping_area) :
        parents = [p1, p2]
        childs = []
        
        for i in range(4) :
            main_set = parents[i//2].copy()
            sub_set = parents[i%2].copy

            child = [" " for _ in range(len(p1))]
            
            ## PMX
            if np.random.random() < 0.4 :
                width = np.random.randint(mapping_area[0], mapping_area[2])
            
            else :
                width = np.random.randint(mapping_area[0], mapping_area[1])
            
            start = np.random.randint(0, mapping_area[2] - width)
            child[start:start+width] = main_set[start:start+width]
            
            ## mapping
            mapping_set = set(main_set[start:start+width]) - set(sub_set[start:start+width])
            
            for t in mapping_set :
                sub = np.where(np.array(sub_set) == t)[0][0]
                
                for k in range(self.cross_area[2]) :
                    sub = np.where(np.array(sub_set) == main_set[sub])[0][0]
                    
                    if (sub < start) | (sub >= start+width) :
                        child[sub] = t
                        break
                
            ## remain set
            current_set = [t for t in child if t != " "]
            remain_set = [t for t in sub_set if t not in current_set]
            
            empty_indx = [i for i, t in enumerate(child) if t == " "]
            
            for k, ind in enumerate(empty_indx) :
                child[ind] = remain_set[k]
                
            childs.append(child)
        
        return childs
    
    def crossover(self, p1, p2, p3, verbs, mutation_chances = 1) :
        structure = [None for _ in range(len(p1))]
        vrbs = [[t for t in p2 if t in verbs], [t for t in p3 if t in verbs]]
        othrs = [[t for t in p2 if t not in verbs], [t for t in p3 if t not in verbs]]
        
        vrb = [" " for _ in range(self.vrbs_area[2])]
        othr = [" " for _ in range(self.othrs_area[2])]
        
        child = ["" for _ in range(len(p1))]
        
        for i, t in enumerate(p1) :
            structure[i] = t in verbs
        
        if mutation_chances > 0 :
            structure = self.mutation_crossover(structure)

        ## crossover verbs
        if self.dupl :
            for i in range(2) :
                vrbs[i] = self.dupl_cleaner(vrbs[i], "labeling")
        

                        
                        
## ------------------------------ 여기서부터 하면 됨 ------------------------------------- 선행되는 부모 인자를 선택, 이후 Combination을 입력하면, 4가지 경우의 수를 모두 헤아려줌.

        
        ## crossover other tokens
        if self.dupl :
            for i in range(2) :
                othrs[i] = self.dupl_cleaner(othrs[i], "labeling")
            
        
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
        
        for _ in range(rep_times) :
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