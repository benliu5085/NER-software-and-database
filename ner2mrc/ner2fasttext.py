import fasttext
import numpy as np
import pandas as pd
import sys
import os
os.system("mkdir -p ../datasets/fasttext")

model = fasttext.load_model("/home/b317l704/mlp/fastText/data/fil9.bin")

for fname in ['train', 'dev', 'test', 'outside']:
    df_input = pd.read_csv(f"DS0/{fname}.tsv", sep='\t', header=None, index_col=None)
    data_out = []
    for i, row in df_input.iterrows():
        words = row[0].split()
        label = row[1].split()
        tmp_ans = []
        tmp_y = [0]*len(words)
        for j, w in enumerate(words):
            tmp_ans.append(model.get_word_vector(w.lower()))
            if label[j] != 'O':
                tmp_y = 1
        data_out.append({"X":np.array(tmp_ans).tolist(), "Y":tmp_y, "ID":row[2]})
    pd.DataFrame(data_out).to_csv(f"../datasets/fasttext/{fname}.csv")
