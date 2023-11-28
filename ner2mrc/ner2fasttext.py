import fasttext
import numpy as np
import pandas as pd
import pickle
import sys
import os
os.system("mkdir -p ../datasets/fasttext")

model = fasttext.load_model("/home/b317l704/mlp/fastText/data/fil9.bin")

for fname in ['train', 'dev', 'test', 'outside', 'test+', 'outside+']:
    df_input = pd.read_csv(f"DS0/{fname}.tsv", sep='\t', header=None, index_col=None)
    data_x = [] # (samples, seq_len, features)
    data_y = [] # (samples, seq_len, 1)
    for i, row in df_input.iterrows():
        words = row[0].split()
        label = row[1].split()
        tmp_x = []
        tmp_y = [0]*len(words)
        for j, w in enumerate(words):
            tmp_x.append(model.get_word_vector(w.lower()))
            if label[j] != 'O':
                tmp_y[j] = 1
        data_x.append(np.array(tmp_x))
        data_y.append(np.array(tmp_y).reshape(-1, 1))
    pickle.dump({'X':data_x, "Y":data_y}, open(f"../datasets/fasttext/{fname}.pkl", 'wb'))
