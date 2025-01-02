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

df_sample = pd.read_csv("sample_submission.csv")

starter = {
    0 : "reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament",
    1 : "reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament walk jump drive bake give sleep laugh the night and",
    2 : "yuletide cheer gifts holiday carol decorations magi polar grinch nutcracker sleigh chimney workshop stocking holly ornament jingle beard naughty nice",
    3 : "yuletide cheer holiday carol gifts decorations polar grinch nutcracker magi sleigh chimney workshop stocking holly ornament jingle beard naughty nice sing relax eat cheer unwrap visit of the and is",
    4 : "poinsettia candle snowglobe peppermint hohoho eggnog chocolate fruitcake candy puzzle doll toy game workshop wrapping bow card greeting season joy merry peace hope believe dream wonder angel cookie milk wreath night star fireplace wish paper the to and of in have not as you that it with we from kaggle",
    5 : "reindeer mistletoe elf gingerbread family advent scrooge chimney fireplace ornament walk jump drive bake give sleep laugh the night and yuletide cheer gifts holiday carol decorations magi polar grinch nutcracker sleigh chimney workshop stocking holly ornament jingle beard naughty nice sing relax eat cheer unwrap visit of the and is poinsettia candle snowglobe peppermint hohoho eggnog chocolate fruitcake candy puzzle doll toy game workshop wrapping bow card greeting season joy merry peace hope believe dream wonder angel cookie milk wreath night star fireplace wish paper the to and of in have not as you that it with we from kaggle"
}

evaluatr = PerplexityCalculator("google/gemma-2-9b")
all_tokens = [l.split() for l in list(starter.values())]

best_word = []

for i, tokens in enumerate(all_tokens) :
    words = [tokens[0]]
    token_list = tokens[1:]
    
    for j in range(len(tokens)-1) :
        word_list = []
        additived_words = []
        
        for t in token_list :
            for k in range(len(words)+1) :
                words.insert(k, t)
                word_list.append(" ".join(words))
                
                del words[k]
                
                additived_words.append(t)
            
        perplexities = evaluatr.get_perplexity(word_list, batch_size = 1024)
        min_index = np.argmin(perplexities)
        words = word_list[min_index].split()
        
        token_list.remove(additived_words[min_index])
        
    best_word.append(words)
    
    
df_submission = pd.DataFrame({"id":[0,1,2,3,4,5], "text":[" ".join(w) for w in best_word]})
df_submission.to_csv("greedy2.csv")

os.system("kaggle competitions submit -c santa-2024 -f greedy2.csv -m .")