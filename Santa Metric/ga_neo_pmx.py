import pandas as pd
import pickle

from algorithm import genetic_neo_pmx

sample = pd.read_csv("sample_submission.csv").text.to_list()[5]
verbs = ["walk", "give", "jump", "drive", "bake", "sleep", "laugh", "sing", "eat", "visit", "relax", "unwrap", "believe", "dream", "hope", "wish", "wrap", "decorate", "play", "wonder", "is", "have"]

public = [['the', 'grinch', 'eat', 'not', 'laugh', 'you', 'scrooge', 'and', 'hohoho',
  'yuletide', 'greeting', 'merry', 'unwrap', 'give', 'as', 'we', 'the', 'of', 'to',
  'from', 'toy', 'doll', 'nutcracker', 'ornament', 'snowglobe', 'game', 'puzzle',
  'card', 'advent', 'candle', 'wreath', 'candy', 'fruitcake', 'bake',
  'gingerbread', 'cookie', 'peppermint', 'chocolate', 'milk', 'and', 'eggnog',
  'sleigh', 'drive', 'reindeer', 'jump', 'polar', 'beard', 'elf', 'workshop',
  'workshop', 'naughty', 'and', 'nice', 'wrapping', 'paper', 'bow', 'poinsettia',
  'holly', 'mistletoe', 'jingle', 'relax', 'have', 'family', 'peace', 'joy',
  'hope', 'wish', 'dream', 'believe', 'star', 'wonder', 'night', 'magi', 'visit',
  'with', 'gifts', 'in', 'stocking', 'chimney', 'fireplace', 'chimney',
  'fireplace', 'angel', 'sleep', 'walk', 'carol', 'sing', 'holiday',
  'decorations', 'ornament', 'night', 'kaggle', 'of', 'the', 'that', 'it', 'is',
  'season', 'cheer', 'cheer'],
 ['the', 'grinch', 'eat', 'not', 'laugh', 'you', 'scrooge', 'and', 'hohoho',
  'yuletide', 'greeting', 'merry', 'unwrap', 'give', 'as', 'we', 'the', 'of', 'to',
  'from', 'toy', 'doll', 'nutcracker', 'ornament', 'snowglobe', 'game', 'card',
  'puzzle', 'advent', 'candle', 'wreath', 'candy', 'fruitcake', 'bake',
  'gingerbread', 'cookie', 'peppermint', 'chocolate', 'milk', 'and', 'eggnog',
  'sleigh', 'drive', 'reindeer', 'jump', 'polar', 'beard', 'elf', 'workshop',
  'workshop', 'naughty', 'and', 'nice', 'wrapping', 'paper', 'bow', 'poinsettia',
  'holly', 'mistletoe', 'jingle', 'relax', 'have', 'family', 'peace', 'joy',
  'hope', 'dream', 'wish', 'believe', 'wonder', 'night', 'magi', 'star', 'visit',
  'with', 'gifts', 'in', 'stocking', 'chimney', 'fireplace', 'chimney',
  'fireplace', 'angel', 'sleep', 'walk', 'carol', 'sing', 'holiday',
  'decorations', 'ornament', 'night', 'kaggle', 'of', 'the', 'that', 'it', 'is',
  'season', 'cheer', 'cheer'],
 ['the', 'grinch', 'eat', 'not', 'laugh', 'you', 'scrooge', 'and', 'hohoho',
  'yuletide', 'greeting', 'merry', 'unwrap', 'give', 'as', 'we', 'the', 'of', 'to',
  'from', 'toy', 'doll', 'nutcracker', 'ornament', 'snowglobe', 'game', 'card',
  'puzzle', 'advent', 'candle', 'wreath', 'candy', 'fruitcake', 'bake',
  'gingerbread', 'cookie', 'peppermint', 'chocolate', 'milk', 'and', 'eggnog',
  'sleigh', 'drive', 'reindeer', 'jump', 'polar', 'beard', 'elf', 'workshop',
  'workshop', 'naughty', 'and', 'nice', 'wrapping', 'paper', 'bow', 'poinsettia',
  'holly', 'mistletoe', 'jingle', 'relax', 'have', 'family', 'peace', 'joy',
  'hope', 'wish', 'dream', 'believe', 'star', 'wonder', 'night', 'magi', 'visit',
  'with', 'gifts', 'in', 'stocking', 'chimney', 'fireplace', 'chimney',
  'fireplace', 'angel', 'sleep', 'walk', 'carol', 'sing', 'holiday',
  'decorations', 'ornament', 'night', 'kaggle', 'of', 'the', 'that', 'it', 'is',
  'season', 'cheer', 'cheer'],
 ['the', 'grinch', 'eat', 'not', 'laugh', 'you', 'scrooge', 'and', 'hohoho',
  'yuletide', 'greeting', 'merry', 'unwrap', 'give', 'as', 'we', 'the', 'of', 'to',
  'from', 'toy', 'doll', 'nutcracker', 'ornament', 'snowglobe', 'game', 'card',
  'puzzle', 'advent', 'candle', 'wreath', 'candy', 'fruitcake', 'bake',
  'gingerbread', 'cookie', 'peppermint', 'chocolate', 'milk', 'and', 'eggnog',
  'sleigh', 'drive', 'reindeer', 'jump', 'polar', 'beard', 'elf', 'workshop',
  'workshop', 'naughty', 'and', 'nice', 'wrapping', 'paper', 'bow', 'poinsettia',
  'holly', 'mistletoe', 'jingle', 'relax', 'have', 'family', 'peace', 'joy',
  'hope', 'believe', 'wish', 'dream', 'star', 'wonder', 'night', 'magi', 'visit',
  'with', 'gifts', 'in', 'stocking', 'chimney', 'fireplace', 'chimney',
  'fireplace', 'angel', 'sleep', 'walk', 'carol', 'sing', 'holiday',
  'decorations', 'ornament', 'night', 'kaggle', 'of', 'the', 'that', 'it', 'is',
  'season', 'cheer', 'cheer']]

optimizr = genetic_neo_pmx(sample, verbs, initial_times = 10000, max_stack = 8, cross_size = 64, mutation_chances = 2, dupl = True, crossover_method = "mixture", parents_size = 8, elite_size = 4, batch_size = 128, public_sample = [" ".join(sample) for sample in public])
best_genome = optimizr.reputation(rep_times = 100)

with open("sample5_neo.pkl", "wb") as f :
    pickle.dump(best_genome, f)