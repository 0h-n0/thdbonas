import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.nn as geonn

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


class GCNSurrogateModel(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 1,
                 hidden_dim: int = 64,
                 n_train_epochs: int = 100):
        super(GCNSurrogateModel, self).__init__()
        self.gcn1 = geonn.GCNConv(input_dim, 16)
        self.gcn2 = geonn.GCNConv(16, 32)
        self.gcn3 = geonn.GCNConv(32, hidden_dim)
        self.activation = nn.Tanh()
        self.last_layer = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.n_train_epochs = n_train_epochs

    def __call__(self, x, edge_index):
        x = self.partial_forward(x, edge_index)
        return self.last_layer(x)

    def partial_forward(self, x, edge_index):
        x = self.activation(self.gcn1(x, edge_index))
        x = self.activation(self.gcn2(x, edge_index))
        x = self.activation(self.gcn3(x, edge_index))
        batch = torch.zeros(x.shape[0]).long()
        x = geonn.global_max_pool(x, batch)
        return x

    def learn(self,
              xtrain=typing.List[Trial],
              ytrain=typing.List[float],
              n_epochs: int = None):
        self.train()
        if n_epochs is None:
            n_epochs = self.n_train_epochs
        # FIXME: should use batch?
        for _ in range(n_epochs):
            for x, y in zip(xtrain, ytrain):
                _, (edge_index, features) = x.graph
                features = features
                x = torch.FloatTensor(features)
                edge_index = torch.LongTensor(edge_index).t()
                edge_index = edge_index
                y = torch.FloatTensor([[y,],])
                self.optimizer.zero_grad()
                out = self(x, edge_index).squeeze()
                loss = F.mse_loss(y, out)
                loss.backward()
                self.optimizer.step()

        bases = []
        with torch.no_grad():
            for x, y in zip(xtrain, ytrain):
                _, (edge_index, features) = x.graph
                features = features
                x = torch.FloatTensor(features)
                edge_index = torch.LongTensor(edge_index).t()
                edge_index = edge_index
                y = torch.FloatTensor([[y,],])
                _base = self.partial_forward(x, edge_index)
                bases.append(_base)
        bases = torch.cat(bases, dim=0)
        return bases

    def predict(self, xeval=typing.List[Trial]):
        bases = []
        with torch.no_grad():
            for x in xeval:
                _, (edge_index, features) = x.graph
                features = features
                x = torch.FloatTensor(features)
                edge_index = torch.LongTensor(edge_index).t()
                edge_index = edge_index
                _base = self.partial_forward(x, edge_index)
                bases.append(_base)
        bases = torch.cat(bases, dim=0)
        return bases
