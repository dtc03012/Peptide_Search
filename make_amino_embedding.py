from gensim.models import Word2Vec
import pandas as pd
import numpy as np


database = pd.read_excel('database.xlsx',sheet_name='Sheet1')

total_word = []
for i in database["Sequence"]:
    now_word = []
    now_word.append("start")
    for p in i:
        now_word.append(p)
    now_word.append("end")
    total_word.append(now_word)
model = Word2Vec(sentences=total_word, size=512, window=5, min_count=0, workers=4, sg=0)

model.wv.save_word2vec_format('amino_w2v')
