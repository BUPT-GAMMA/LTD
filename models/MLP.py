import torch
import torch.nn as nn
import torch.nn.functional as F
from data.utils import normalize_adj


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer).
            If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True       # default is linear model
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=1-dropout)

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model ()
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:  # If linear model
            # return F.log_softmax(self.linear(x), dim=1)
            return self.linear(x)
        else:                   # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](self.dropout(h)))
            return F.sigmoid(self.linears[self.num_layers - 1](self.dropout(h)))

