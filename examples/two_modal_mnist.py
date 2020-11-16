#!/usr/bin/env python
import time
import copy
from collections import OrderedDict
import warnings

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torchex.nn as exnn
import networkx as nx


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(392, 32)
        self.l2 = nn.Linear(392, 32)        
        self.l3 = nn.Linear(32, 32)
        self.l4 = nn.Linear(128, 128)
        self.l5 = nn.Linear(128, 10)        

    def forward(self, x1, x2):
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x1.size(0), -1)
        x1 = F.relu(self.l1(x1))
        x1_ = self.l3(x1)        
        x2 = F.relu(self.l2(x2))
        x1 = torch.cat([x1, x1_], dim=1)
        x2 = torch.cat([x2, x1_], dim=1)
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.l4(x))
        x = F.relu(x)
        x = self.l5(x)
        return x


def objective():
    model = Model()
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    model = model.to(device)
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

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(10):
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            d1, d2 = torch.split(data, 14, dim=2)
            optimizer.zero_grad()
            output = model(d1, d2)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            output = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            total_loss += loss.detach().cpu().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = 100. * correct / ((batch_idx + 1) * batch_size)
        print('>>>', acc, total_loss)
        
    model.eval()
    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        d1, d2 = torch.split(data, 14, dim=2)
        output = model(d1, d2)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        output = F.softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        total_loss += loss.detach().cpu().item()
        correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / ((batch_idx + 1) * batch_size)
    print('test, ', acc)
    return acc

if __name__ == "__main__":
    objective()
