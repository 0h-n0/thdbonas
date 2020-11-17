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
from frontier_graph import NetworkxInterface

from inferno.extensions.layers.reshape import Concatenate
from inferno.extensions.containers import Graph

from thdbonas import Searcher, Trial

from blocks import BlockGenerator
from network_generator.frame_generator import FrameGenerator
from network_generator.module_generator import NNModuleGenerator
from network_generator.output_size_searcher import OutputSizeSearcher

np.random.seed(0)


def objectve(trial):
    model, (edges, _) = trial.graph
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

    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
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
        if batch_idx > 20:
            break

    acc = 100. * correct / ((batch_idx + 1) * batch_size)
    print('>>>', acc, total_loss)
    return acc


def get_subgraphs(frame_graph, starts, ends, sample_size):
    fg = FrameGenerator(frame_graph, starts, ends)
    n_nodes = len(frame_graph.nodes())    
    network_input_sizes = {v: 7 for v in starts}
    network_output_sizes = {v: 10 for v in ends}
    kernel_sizes = [1,]
    strides = [1,]
    output_channel_candidates = [32, 64, 128, 256]
    samples = []
    for idx in range(sample_size):
        frame = fg.sample_graph()
        oss = OutputSizeSearcher(frame, starts, ends, max(network_input_sizes.values()), True, kernel_sizes, strides)
        # あまりに単純な場合は省く
        if len(oss.g_compressed.nodes) <= 3: continue

        output_sizes = []
        for _ in range(100):
            output_dimensions = oss.sample_output_dimensions()
            result = oss.sample_valid_output_size(network_input_sizes, output_dimensions)
            if result == False: break
            else: output_sizes.append((result, output_dimensions))

        if len(output_sizes) == 0: continue

        opt_sizes, opt_dims =\
            max(output_sizes, key=lambda x: len(set(x[0].values())) * (max(x[0].values()) - min(x[0].values())))
        mg = NNModuleGenerator(frame, starts, ends, network_input_sizes, opt_sizes, opt_dims, network_output_sizes,
                               kernel_sizes, strides, output_channel_candidates, n_nodes)

        module, node_features = mg.run()
        node_features = np.array(node_features)
        edges = [list(e) for e in frame.edges]
        samples.append((module, (edges, node_features)))
    return samples


def one_round(n_elongation: int):
    generator = BlockGenerator(2, 1)
    for _ in range(n_elongation):
        generator.elongate()
    starts = generator.start_nodes
    ends = generator.end_nodes
    samples = get_subgraphs(generator.graph, starts, ends, 5000)
    searcher = Searcher()
    searcher.register_trial('graph', samples)
    num_node_features = 9 #node_features.shape[1]
    n_trials = 50
    n_random_trials = 10
    model_kwargs = dict(
        input_dim=num_node_features,
        n_train_epochs=400,
    )
    result = searcher.search(objectve,
                             n_trials=n_trials,
                             deep_surrogate_model=f'thdbonas.deep_surrogate_models:GCNSurrogateModel',
                             n_random_trials=n_random_trials,
                             model_kwargs=model_kwargs)
    return result

    
if __name__ == "__main__":
    best_acc = 0
    results = []
    for i in range(4):
        print(f'>>>>> {i}th round')
        result = one_round(i)
        if result.best_value > best_acc:
            results.append(result)
            best_acc = float(result.best_value)
            print(result.best_trial)        
            print(f'result.best_value = {result.best_value}')
        else:
            print(f'break {best_acc}')
            print(result.best_trial)        
            break
    print(results[-1])
    print(results[-2])    

