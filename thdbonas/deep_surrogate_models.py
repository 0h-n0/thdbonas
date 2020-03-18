import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .trial import Trial


class SimpleNetwork(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 n_train_epochs: int = 100):
        super(SimpleNetwork, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, hidden_dim),
            nn.Tanh())
        self.last_layer = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.n_train_epochs = n_train_epochs

    def __call__(self, x):
        return self.last_layer(self.first_layer(x))

    def partial_forward(self, x):
        return self.first_layer(x)

    def learn(self,
              xtrain=typing.List[Trial],
              ytrain=typing.List[float],
              n_epochs: int = None):
        self.train()
        if n_epochs is None:
            n_epochs = self.n_train_epochs
        for _ in range(n_epochs):
            for x, y in zip(xtrain, ytrain):
                x = torch.FloatTensor(x.to_numpy())
                y = torch.FloatTensor([[y,],])
                self.optimizer.zero_grad()
                out = self(x).squeeze()
                loss = F.mse_loss(y, out)
                loss.backward()
                self.optimizer.step()
        x = torch.FloatTensor([x.to_numpy() for x in xtrain])
        with torch.no_grad():
            bases = self.partial_forward(x)
        return bases

    def predict(self, xeval=typing.List[Trial]):
        x = torch.FloatTensor([x.to_numpy() for x in xeval])
        with torch.no_grad():
            bases = self.partial_forward(x)
        return bases
