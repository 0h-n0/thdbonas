#!/usr/bin/env python
import uuid
import typing
from pathlib import Path

import torch
import torch.nn as nn
import torchex.nn as exnn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from thdbonas import Searcher, Trial


class Model(nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.model = nn.ModuleList([l[0] for l in layers])

    def forward(self, x):
        for m in self.model:
            x = m(x)
        x = F.log_softmax(x, dim=1)
        return x


def objectve(trial):
    layers, _ = trial.graph
    use_cuda = True

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Model(layers).to(device)
    lr = 0.01
    batch_size = 128

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = Model(layers).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    model.train()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        total_loss += loss.detach().cpu().item()
        correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / (len(train_loader) * batch_size)
    print(acc)
    return acc


if __name__ == '__main__':
    m = MultiHeadLinkedListLayer()
    # graph created
    kwargs = [dict(out_features=i) for i in range(16, 129, 32)]
    m.append_lazy(exnn.Flatten, [dict(),])
    m.append_lazy(exnn.Linear, kwargs)
    m.append_lazy(nn.ReLU, [dict(),])
    m.append_lazy(exnn.Linear, kwargs)
    m.append_lazy(nn.ReLU, [dict(),])
    m.append_lazy(exnn.Linear, kwargs)
    m.append_lazy(nn.ReLU, [dict(),])
    m.append_lazy(exnn.Linear, [dict(out_features=10),])
    g = Generator(m, dump_nn_graph=True, sparse_graph=True)
    num_nodes = 10
    num_layer_type = 3
    searcher = Searcher()
    searcher.register_trial('graph', g)
    n_trials = 30
    model_kwargs = dict(
        input_dim=num_layer_type,
        n_train_epochs=400,
    )
    _ = searcher.search(objectve,
                        n_trials=n_trials,
                        deep_surrogate_model=f'thdbonas.deep_surrogate_models:GCNSurrogateModel',
                        n_random_trials=10,
                        model_kwargs=model_kwargs)
    print(searcher.result)
    print('best_trial', searcher.best_trial)
    print('best_value', searcher.best_value)
