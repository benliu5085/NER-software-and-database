import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    field_size = len(batch[0])
    output = []
    for i in range(field_size):
        output.append(pad_sequence([x[i] for x in batch], batch_first=True, padding_value=0))
    return output

def collateFT(batch):
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []
    for field_idx in range(2):
        print()
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)
    return output

class bioNerDSDataset(Dataset):
    def __init__(self, fname):
        all_data = pd.read_csv(fname)
        X_train, Y_train = all_data.iloc[:,:7], all_data.iloc[:,8:9]
        self.X = torch.tensor(X_train.values, dtype=torch.float32)
        self.Y = torch.tensor(Y_train.values, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        return [self.X[item], self.Y[item]]

class fasttextDataset(Dataset):
    def __init__(self, fname):
        all_data = pickle.load(open(fname, 'rb'))
        self.X = all_data['X']
        self.Y = all_data['Y']
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        return [torch.tensor(self.X[item], dtype=torch.float32),
                torch.tensor(self.Y[item], dtype=torch.float32)]

class tfidfDataset(Dataset):
    def __init__(self, fname):
        all_data = pickle.load(open(fname, 'rb'))
        self.X = torch.tensor(all_data['X_data'], dtype=torch.float32)
        self.Y = torch.tensor((all_data['Y_data'] == 1).sum(axis=1).reshape(-1, 1), dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, item):
        return [self.X[item], self.Y[item]]

def run_dataset():
    from torch.utils.data import DataLoader
    from model import MLPModel
    getInputSize = {'bioNerDS':7, 'fasttext':100, 'TFIDF':13047}

    dataset = bioNerDSDataset(fname="datasets/bioNerDS/train.csv")
    dataloader = DataLoader(dataset=dataset,  batch_size=4, shuffle=False)
    for batch in dataloader:
        X_train, Y_train = batch
        print(X_train.shape, Y_train.shape)

    dataset = tfidfDataset(fname="datasets/TFIDF/train.pkl")
    dataloader = DataLoader(dataset=dataset,  batch_size=4, shuffle=False, collate_fn=collate)
    for batch in dataloader:
        X_train, Y_train = batch
        print(X_train.shape, Y_train.shape)

    dataset = fasttextDataset(fname="datasets/fasttext/train.pkl")
    dataloader = DataLoader(dataset=dataset,  batch_size=4, shuffle=False, collate_fn=collate)
    for batch in dataloader:
        X_train, Y_train = batch
        print(X_train.shape, Y_train.shape)

if __name__ == '__main__':
    run_dataset()
