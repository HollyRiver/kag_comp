import numpy as np
import itertools
from algorithm.kaggle_evaluate import PerplexityCalculator
from typing import List

class genetic_brute :
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
        mutation_chances = 2,
        dupl = False,
        crossover_method = "mixture",
        parents_size = 10,
        elite_size = 3,
        batch_size = 1024,
        load_in_8bit = False,
        public_sample = None) :
        
        
        ## create evaluator
        self.batch_size = batch_size
        self.load_in_8bit = load_in_8bit
        self.evaluatr = PerplexityCalculator("google/gemma-2-9b", load_in_8bit = self.load_in_8bit)
        
        ## first genomes generate
        self.genomes = ["" for _ in range(initial_times)]
                
        for i in range(initial_times) :
            self.genomes[i] = np.random.permutation(sample.split())
        
        if public_sample != None :
            for sample in public_sample :
                self.genomes.append(sample.split())
        
        self.genomes = np.array(self.genomes) ## numpy array로 변경
        
        ## parameter setting
        self.max_stack = max_stack
        self.stack = 0  ## current stack
        
        self.mutation_chances = mutation_chances    ## mutation chance in crossover
        self.cross_size = cross_size    ## crossover sample size
        self.othrs_area = [(len(sample.split()) - len(verbs))//(2**(2-i)) for i in range(3)]
        self.parents_size = parents_size
        self.crossover_method = crossover_method
        self.vrbs = verbs
        self.dupl = dupl
        
        ## verbs setting
        self.vrbs_set = list(itertools.permutations(verbs)) ## v! setting
        
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
    
    
    def dupl_cleaner(self, p, clean_type) :
        """
        for handling duplication (대충 씀)

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
        """_summary_
        부모 둘을 넣으면 PMX 방식으로 순열교차 후 한 개의 자녀를 리턴

        Args:
            p1 (List[str]): 부모 1
            p2 (List[str]): 부모 2
            mapping_area (List[a, b, c]): 세 개의 정수로 이뤄진 리스트, 각각 최소 교차 영역, 최대 교차 영역, 조건부 최대 교차 영역

        Returns:
            List[str]: 첫 입력 부모를 메인으로 하는 자녀 유전자 하나 리턴
        """
        main_set = p1
        sub_set = p2
        
        ## PMX
        if np.random.random() < 0.4 :
            width = np.random.randint(mapping_area[0], mapping_area[2])
        
        else :
            width = np.random.randint(mapping_area[0], mapping_area[1])
        
        start = np.random.randint(0, mapping_area[2] - width)
        child = [" " for _ in range(len(p1))]
        child[start:start+width] = main_set[start:start+width]
        
        ## mapping
        mapping_set = set(sub_set[start:start+width]) - set(main_set[start:start+width])
        
        for t in mapping_set :
            sub = np.where(np.array(sub_set) == t)[0][0]
            
            for k in range(mapping_area[2]) :
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
            
        
        return child
    
    def crossover(self, p1, p2, p3, verbs) :
        structure = [None for _ in range(len(p1))]
        othrs = [[t for t in p2 if t not in verbs], [t for t in p3 if t not in verbs]]
        
        childs = []
        
        for i, t in enumerate(p1) :
            structure[i] = t in verbs

        ## crossover
        if self.dupl :
            for i in range(2) :
                othrs[i] = self.dupl_cleaner(othrs[i], "labeling")
                
        othrs_child = self.PMX(othrs[0], othrs[1], self.othrs_area) ## one set
        
        if self.dupl :
            othrs_child = self.dupl_cleaner(othrs_child, "cleaning")
        
        ## merge childs : v! times
        for verb in self.vrbs_set :
            child = ["" for _ in range(len(p1))]
            
            a = 0
            b = 0
            
            for j, s in enumerate(structure) :
                othrs = othrs_child
                
                if s :
                    child[j] = verb[a]
                    a += 1
                else :
                    child[j] = othrs[b]
                    b += 1
                
            childs.append(child)
        
        return childs ## return v! childs
    

    def mutation(self, genome) :
        origin = np.array(genome)
        lnth = len(origin)
        
        for _ in range(self.mutation_chances) :
            ## mutate randomly
            if np.random.random() > 0.75 :
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
        cross_size = self.cross_size
        genome_set = self.genomes   ## initial genomes : numpy.array
        best_genome = self.best_genome
        best_perplexties = [9999 for _ in range(self.parents_size)]
        parents_indx = self.parents_indx
        
        for _ in range(rep_times) :
            pairs = []
            
            ## generate parents sets
            for main_set in parents_indx :
                pair_parents = []
                
                for structure in parents_indx :
                    for sub_set in parents_indx :
                        if main_set != sub_set :
                            pair_parents.append([structure, main_set, sub_set])
                            
                pairs.append(pair_parents)
            
            child_set = []
            
            
            ## crossover & mutation
            for pair_parents in pairs :
                childs = []
                
                for pair in pair_parents :
                    parents_genome = [genome_set[idx] for idx in pair]
                    
                    for _ in range(cross_size) :
                        crossover_genomes = self.crossover(parents_genome[0], parents_genome[1], parents_genome[2], verbs = self.vrbs)
                        for genome in crossover_genomes :
                            childs.append(self.mutation(genome))
                            
                child_set.append(childs)

            
            ## setting new genome set
            genome_parts = [np.unique([" ".join(genome) for genome in childs]) for childs in child_set]
            set_size = np.cumsum([0] + [len(gs) for gs in genome_parts])
            
            genome_set = np.array([])
            
            for part in genome_parts :
                genome_set = np.concat([genome_set, part])
                

            ## evaluate
            self.evaluatr.clear_gpu_memory()
            self.evaluatr = PerplexityCalculator("google/gemma-2-9b")
            
            perplexities = np.array(self.evaluatr.get_perplexity(genome_set, batch_size = self.batch_size))
            
            ## renewal
            if perplexities.min() < best_genome[1] :
                best_genome = [genome_set[perplexities.argmin()], perplexities.min()]
                print(f"best genome : {best_genome}")
                stack = 0
                
            else :
                stack += 1
                
                if stack >= self.max_stack :
                    break
            
            for i, best_in_set in enumerate(best_perplexties) :                
                if perplexities[set_size[i]:set_size[i+1]].min() < best_in_set :
                    best_perplexties[i] = perplexities[set_size[i]:set_size[i+1]].min()
            
            ## select parents
            for i in range(self.parents_size) :
                parents_indx[i] = set_size[i] + perplexities[set_size[i]:set_size[i+1]].argmin()
                print(perplexities[set_size[i]:set_size[i+1]].min())

            ## genome set formatting
            genome_set = [genome.split() for genome in genome_set]
            
            print(f"parents perplexities : {perplexities[parents_indx]}")
            print(np.array(genome_set)[parents_indx[:2]])
            print(np.array(genome_set)[parents_indx[2:]])
            
        return best_genome