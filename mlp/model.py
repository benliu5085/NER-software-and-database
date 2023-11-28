import torch
import torch.nn as nn
from torch.nn import functional as F

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size=2048, dropout_rate=0.1, act_func="gelu"):
        super().__init__()
        self.classifier1 = nn.Linear(input_size, hidden_size)
        self.classifier2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.act_func = act_func

    def forward(self, input_features):
        features_output1 = self.classifier1(input_features)
        if self.act_func == "gelu":
            features_output1 = F.gelu(features_output1)
        elif self.act_func == "relu":
            features_output1 = F.relu(features_output1)
        elif self.act_func == "tanh":
            features_output1 = F.tanh(features_output1)
        else:
            raise ValueError
        features_output1 = self.dropout(features_output1)
        single_logits = self.classifier2(features_output1)
        return single_logits
