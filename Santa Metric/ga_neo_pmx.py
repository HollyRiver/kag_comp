import pandas as pd
import pickle

from algorithm import genetic_neo_pmx

sample = pd.read_csv("sample_submission.csv").text.to_list()[5]
verbs = ["walk", "give", "jump", "drive", "bake", "sleep", "laugh", "sing", "eat", "visit", "relax", "unwrap", "believe", "dream", "hope", "wish", "wrap", "decorate", "play", "wonder", "is", "have"]

public = [['the','grinch','eat','not','laugh','you','scrooge','hohoho','yuletide',
  'greeting','merry','unwrap','give','cheer','as','we','to','and','the',
  'from','ornament','nutcracker','toy','doll','puzzle','snowglobe','game',
  'night','card','advent','candle','stocking','candy','fruitcake',
  'cookie','gingerbread','bake','peppermint','chocolate','milk','and',
  'eggnog','sleigh','drive','reindeer','walk','polar','beard','elf',
  'workshop','workshop','naughty','and','nice','wrapping','paper',
  'mistletoe','poinsettia','holly','wreath','jingle','jump','joy','peace',
  'family','relax','wish','hope','star','of','wonder','magi','visit',
  'with','gifts','bow','have','believe','in','chimney','fireplace',
  'chimney','fireplace','angel','sleep','dream','carol','sing','holiday',
  'decorations','ornament','night','kaggle','of','the','that','it','is',
  'season','cheer'],
 ['the','grinch','eat','not','laugh','you','scrooge','hohoho','yuletide',
  'greeting','merry','unwrap','give','cheer','we','to','and','the','from',
  'ornament','nutcracker','toy','doll','snowglobe','game','night',
  'puzzle','card','advent','candle','wreath','stocking','candy',
  'fruitcake','cookie','gingerbread','bake','peppermint','chocolate',
  'milk','and','eggnog','sleigh','drive','reindeer','walk','polar',
  'beard','elf','workshop','workshop','naughty','and','nice','wrapping',
  'paper','mistletoe','poinsettia','holly','jingle','jump','joy','peace',
  'family','relax','wish','hope','star','of','wonder','magi','visit',
  'with','gifts','bow','have','believe','in','as','chimney','fireplace',
  'chimney','fireplace','angel','sleep','dream','carol','sing','holiday',
  'decorations','ornament','night','kaggle','of','the','that','it','is',
  'season','cheer'],
 ['the','grinch','eat','not','laugh','you','scrooge','hohoho','yuletide',
  'greeting','merry','unwrap','give','cheer','we','to','and','the','from',
  'ornament','nutcracker','toy','doll','snowglobe','game','night',
  'puzzle','card','advent','candle','wreath','stocking','candy',
  'fruitcake','gingerbread','cookie','bake','peppermint','chocolate',
  'milk','and','eggnog','sleigh','drive','reindeer','walk','polar',
  'beard','elf','workshop','workshop','naughty','and','nice','wrapping',
  'paper','mistletoe','poinsettia','holly','jingle','jump','joy','peace',
  'family','relax','wish','hope','star','of','wonder','magi','visit',
  'with','gifts','bow','have','believe','in','as','chimney','fireplace',
  'chimney','fireplace','angel','sleep','dream','carol','sing','holiday',
  'decorations','ornament','night','kaggle','of','the','that','it','is',
  'season','cheer'],
 ['the','grinch','eat','not','laugh','you','scrooge','hohoho','yuletide',
  'greeting','merry','unwrap','give','cheer','as','we','to','and','the',
  'from','ornament','nutcracker','toy','doll','puzzle','snowglobe','game',
  'night','card','advent','candle','stocking','candy','fruitcake',
  'cookie','gingerbread','bake','peppermint','chocolate','milk','and',
  'eggnog','sleigh','drive','reindeer','visit','polar','beard','elf',
  'workshop','workshop','naughty','and','nice','wrapping','paper',
  'mistletoe','poinsettia','holly','wreath','jingle','jump','joy','peace',
  'family','relax','wish','hope','star','of','wonder','magi','walk',
  'with','gifts','bow','have','believe','in','chimney','fireplace',
  'chimney','fireplace','angel','sleep','dream','carol','sing','holiday',
  'decorations','ornament','night','kaggle','of','the','that','it','is',
  'season','cheer']]

optimizr = genetic_neo_pmx(sample, verbs, initial_times = 10000, max_stack = 8, cross_size = 16, mutation_chances = 1, dupl = True, crossover_method = "mixture", parents_size = 16, elite_size = 4, batch_size = 128, public_sample = [" ".join(sample) for sample in public])
best_genome = optimizr.reputation(rep_times = 100)

with open("sample5_neo.pkl", "wb") as f :
    pickle.dump(best_genome, f)