import pandas as pd
import pickle

from algorithm import genetic_neo_pmx

sample = pd.read_csv("sample_submission.csv").text.to_list()[5]
verbs = ["walk", "give", "jump", "drive", "bake", "sleep", "laugh", "sing", "eat", "visit", "relax", "unwrap", "believe", "dream", "hope", "wish", "wrap", "decorate", "play", "wonder", "is", "have"]

public = [['the', 'grinch', 'eat', 'not', 'you', 'scrooge', 'and', 'hohoho', 'yuletide',
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
  'cheer', 'and', 'cheer'],
 ['the', 'grinch', 'eat', 'not', 'you', 'scrooge', 'and', 'hohoho', 'yuletide',
  'greeting', 'have', 'merry', 'unwrap', 'give', 'as', 'we', 'the', 'of', 'from',
  'to', 'toy', 'doll', 'ornament', 'snowglobe', 'card', 'game', 'puzzle',
  'advent', 'candle', 'wreath', 'cookie', 'bake', 'fruitcake', 'gingerbread',
  'candy', 'peppermint', 'chocolate', 'milk', 'and', 'eggnog', 'sleigh', 'drive',
  'reindeer', 'jump', 'polar', 'kaggle', 'beard', 'elf', 'workshop', 'workshop',
  'naughty', 'nice', 'wrapping', 'paper', 'bow', 'ornament', 'nutcracker',
  'poinsettia', 'holly', 'mistletoe', 'jingle', 'relax', 'family', 'laugh',
  'joy', 'peace', 'sleep', 'dream', 'wish', 'hope', 'believe', 'wonder', 'magi',
  'visit', 'with', 'gifts', 'in', 'stocking', 'chimney', 'fireplace',
  'fireplace', 'chimney', 'star', 'angel', 'night', 'night', 'walk', 'carol',
  'sing', 'holiday', 'decorations', 'that', 'it', 'is', 'the', 'season', 'of',
  'cheer', 'and', 'cheer'],
 ['the', 'grinch', 'eat', 'not', 'you', 'scrooge', 'and', 'hohoho', 'yuletide',
  'greeting', 'have', 'merry', 'unwrap', 'give', 'as', 'we', 'the', 'of', 'from',
  'to', 'toy', 'doll', 'ornament', 'snowglobe', 'card', 'game', 'puzzle',
  'advent', 'candle', 'wreath', 'cookie', 'bake', 'fruitcake', 'gingerbread',
  'candy', 'peppermint', 'chocolate', 'milk', 'and', 'eggnog', 'sleigh', 'drive',
  'reindeer', 'jump', 'polar', 'kaggle', 'beard', 'elf', 'workshop', 'workshop',
  'naughty', 'nice', 'wrapping', 'paper', 'bow', 'ornament', 'nutcracker',
  'poinsettia', 'holly', 'mistletoe', 'jingle', 'relax', 'family', 'laugh',
  'joy', 'peace', 'sleep', 'dream', 'wish', 'hope', 'believe', 'wonder', 'magi',
  'visit', 'with', 'gifts', 'in', 'stocking', 'fireplace', 'fireplace',
  'chimney', 'chimney', 'angel', 'star', 'night', 'night', 'walk', 'carol',
  'sing', 'holiday', 'decorations', 'that', 'it', 'is', 'the', 'season', 'of',
  'cheer', 'and', 'cheer'],
 ['the', 'grinch', 'eat', 'not', 'you', 'scrooge', 'and', 'hohoho', 'yuletide',
  'greeting', 'have', 'merry', 'unwrap', 'give', 'as', 'we', 'the', 'of', 'to',
  'from', 'toy', 'doll', 'ornament', 'snowglobe', 'card', 'game', 'puzzle',
  'advent', 'candle', 'wreath', 'cookie', 'bake', 'fruitcake', 'gingerbread',
  'candy', 'peppermint', 'chocolate', 'milk', 'and', 'eggnog', 'sleigh', 'drive',
  'reindeer', 'jump', 'polar', 'kaggle', 'beard', 'elf', 'workshop', 'workshop',
  'naughty', 'nice', 'wrapping', 'paper', 'bow', 'ornament', 'nutcracker',
  'poinsettia', 'holly', 'mistletoe', 'jingle', 'relax', 'family', 'laugh',
  'joy', 'peace', 'sleep', 'dream', 'wish', 'hope', 'believe', 'wonder', 'magi',
  'visit', 'with', 'gifts', 'in', 'stocking', 'fireplace', 'fireplace',
  'chimney', 'chimney', 'angel', 'star', 'night', 'night', 'walk', 'carol',
  'sing', 'holiday', 'decorations', 'that', 'it', 'is', 'the', 'season', 'of',
  'cheer', 'and', 'cheer']]

optimizr = genetic_neo_pmx(sample, verbs, initial_times = 10000, max_stack = 8, cross_size = 64, mutation_chances = 2, dupl = True, crossover_method = "mixture", parents_size = 8, elite_size = 4, batch_size = 128, public_sample = [" ".join(sample) for sample in public])
best_genome = optimizr.reputation(rep_times = 100)

with open("sample5_neo.pkl", "wb") as f :
    pickle.dump(best_genome, f)