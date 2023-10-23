from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import pickle
import os
os.system("mkdir -p ../datasets/TFIDF")
vectorizer = TfidfVectorizer()

dfs = {}
for fname in ['train', 'dev', 'test', 'outside']:
    df_input = pd.read_csv(f"DS0/{fname}.tsv", sep='\t', header=None, index_col=None)
    df_input.columns = ['context', 'label', 'sid']
    dfs[fname] = df_input


corpus = dfs['train']['context']
vectorizer.fit(corpus)
vocab_ = vectorizer.get_feature_names()

# sanity check, you should see nothing on standard output
for ss in ['train', 'dev', 'test', 'outside']:
    for i, row in dfs[ss].iterrows():
        words = row['context'].split()
        label = row['label'].split()
        if len(words) != len(label):
            print(f"{ss}: mismatch length!")

for ss in ['train', 'dev', 'test', 'outside']:
    df_input = dfs[ss]
    Y_data = []
    X_data = []
    for i, row in df_input.iterrows():
        tmp_x = vectorizer.transform([row['context']]).toarray()
        tmp_y = np.zeros(tmp_x.shape)
        words = row['context'].split()
        label = row['label'].split()
        for j, ll in enumerate(label):
            if ll != 'O':
                if words[j].lower() in vocab_:
                    tmp_y[0, vocab_.index(words[j].lower())] = 1
        X_data.append(tmp_x)
        Y_data.append(tmp_y)
    data = {}
    data['Y_data']  = np.squeeze(np.array(Y_data))
    data['X_data']  = np.squeeze(np.array(X_data))
    data['row_index'] = dfs[ss]['sid'].tolist()
    data['col_index'] = vocab_
    pickle.dump(data, open(f"../datasets/TFIDF/{ss}.pkl", 'wb'))
