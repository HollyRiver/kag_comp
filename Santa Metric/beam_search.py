import gc ## for memory
import os
import copy
from math import exp
from collections import Counter
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import transformers
import torch

from kaggle_evaluate import PerplexityCalculator


evaluatr = PerplexityCalculator("google/gemma-2-9b")
greedy_starter = pd.read_csv("greedy2.csv", index_col = 0)

all_tokens = greedy_starter.text.str.split().to_list()
beam_size = [5, 10, 10, 15, 15, 15]
best_word = []

for i, tokens in enumerate(all_tokens) :
    token_start = tokens[0]
    width = beam_size[i]
    token_list = [copy.deepcopy(tokens[1:]) for _ in range(width)]
    beam_set = []
    
    for j in range(len(tokens)-1) :
        word_list = []
        
        if beam_set == [] :            
            for t in token_list[0] :
                word_list.append(" ".join([token_start] + [t]))
            
            perplexities = np.array(evaluatr.get_perplexity(word_list, batch_size = 1024))
            min_indexs = np.where(perplexities.argsort() < width)[0]
            token_len = len(token_list[0])
            beam_set = [[token_start, token_list[min_index//token_len][min_index%token_len]] for min_index in min_indexs]
        
            for l in range(width) :
                del token_list[l][min_indexs[l]%token_len]
            
        else :
            for m, beam in enumerate(beam_set) :
                for t in token_list[m] :
                    word_list.append(" ".join(beam + [t]))
                
            perplexities = np.array(evaluatr.get_perplexity(word_list, batch_size = 1024))
            min_indexs = np.where(perplexities.argsort() < width)[0]
            token_len = len(token_list[0])
            tmp = copy.deepcopy(beam_set)
            beam_set = [tmp[min_index//token_len] + [token_list[min_index//token_len][min_index%token_len]] for min_index in min_indexs]
        
            tmp = copy.deepcopy(token_list)
            
            for l, min_index in enumerate(min_indexs) :
                token_list[l] = copy.deepcopy(tmp[min_index//token_len])
                del token_list[l][min_index%token_len]
                

    print(f"cycle rooped")
        
    best_word.append(beam_set[perplexities.argmax()])
    

df_submission = pd.DataFrame({"id":[0,1,2,3,4,5], "text":[" ".join(w) for w in best_word]})
df_submission.iloc[0, 1] = "reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament"
df_submission.to_csv("beam.csv")

os.system("kaggle competitions submit -c santa-2024 -f beam.csv -m .")