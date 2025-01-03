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
df_sample = pd.read_csv("sample_submission.csv", index_col = 0)

all_tokens = df_sample.text.str.split().to_list()
beam_size = [5, 10, 10, 15, 25, 50]
best_word = []

for i, tokens in enumerate(all_tokens) :
    width = beam_size[i]
    token_list = [copy.deepcopy(tokens) for _ in range(width)]
    beam_set = [[tok] for tok in copy.deepcopy(tokens)]
    
    for j in range(len(tokens)) :
        word_list = []
        
        if j == 0 :            
            for t in token_list[0] :
                word_list.append(t)
            
            perplexities = np.array(evaluatr.get_perplexity(word_list, batch_size = 256))
            
        else :
            for m, beam in enumerate(beam_set) :
                for t in token_list[m] :
                    word_list.append(" ".join(beam + [t]))
                    
            if len(word_list[0]) > 40 :
                perplexities = np.array(evaluatr.get_perplexity(word_list, batch_size = 128))
                
            else :
                perplexities = np.array(evaluatr.get_perplexity(word_list, batch_size = 256))
                
        min_indexs = perplexities.argsort()[:width]
        token_len = len(token_list[0])
        tmp = copy.deepcopy(beam_set)
        beam_set = [tmp[min_index//token_len] + [token_list[min_index//token_len][min_index%token_len]] for min_index in min_indexs]
    
        tmp = copy.deepcopy(token_list)
        
        for l, min_index in enumerate(min_indexs) :
            token_list[l] = copy.deepcopy(tmp[min_index//token_len])
            del token_list[l][min_index%token_len]
                
        
    best_word.append(beam_set[perplexities.argmin()])
    

df_submission = pd.DataFrame({"id":[0,1,2,3,4,5], "text":[" ".join(w) for w in best_word]})
df_submission.iloc[0, 1] = "reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament"
df_submission.to_csv("beam.csv")

os.system("kaggle competitions submit -c santa-2024 -f beam.csv -m .")