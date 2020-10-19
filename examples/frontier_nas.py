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

np.random.seed(0)


def conv2d(out_channels, kernel_size, stride):
    conv = nn.Sequential(
        exnn.Conv2d(out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride),
        nn.ReLU())
    return conv


class FlattenLinear(nn.Module):
    def __init__(self, out_channels, activation='relu'):
        super(FlattenLinear, self).__init__()
        self.out_channels = out_channels
        self.linear = exnn.Linear(self.out_channels)
        self.flatten = exnn.Flatten()
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x):
        return self.activation(self.linear(self.flatten(x)))

class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
        self.cat = Concatenate()

    def reshape_1d(self, x):
        B = x.size(0)
        return x.view(B, -1)
        
    def forward(self, *inputs):
        reshaped_inputs = []
        for x in inputs:
            reshaped_inputs.append(self.reshape_1d(x))
        return self.cat(*reshaped_inputs)
    

class ModuleGen:
    def __init__(self):
        self.layer = []
        self.layer_dict = OrderedDict()
        self.layer_dict['concat'] = 0
        self._len = None

    def register(self, module_name: str, **params):
        self.layer.append((module_name, params))
        self.layer_dict[module_name] = 0
        for k in params:
            self.layer_dict[k] = 0

    def __getitem__(self, idx):
        module, vec = self.construct(idx)
        return module, vec

    def construct(self, idx):
        _layer_dict = copy.deepcopy(self.layer_dict)
        (module_name, params) = self.layer[idx]
        _layer_dict[module_name] = 1
        for k, v in params.items():
            _layer_dict[k] = v
        vec = list(_layer_dict.values())
        if module_name == 'conv2d':
            return conv2d(**params), vec
        elif module_name == 'linear':
            return FlattenLinear(**params), vec
        elif module_name == 'identity':
            return exnn.Flatten(), vec

    def get_linear(self, out_channels, activation='relu'):
        _layer_dict = copy.deepcopy(self.layer_dict)
        _layer_dict['linear'] = 1
        _layer_dict['out_channels'] = out_channels
        vec = list(_layer_dict.values())
        return FlattenLinear(out_channels, activation), vec

    def get_identity_vec(self):
        _layer_dict = copy.deepcopy(self.layer_dict)
        _layer_dict['identity'] = 1
        vec = list(_layer_dict.values())
        return vec

    def get_cat(self):
        _layer_dict = copy.deepcopy(self.layer_dict)
        _layer_dict['concat'] = 1
        vec = list(_layer_dict.values())
        return Concat(), vec

    def get_empty_mat(self, n_node: int):
        _layer_dict = copy.deepcopy(self.layer_dict)
        n_features = len(_layer_dict.values())
        mat = np.zeros((n_node, n_features))
        return mat

    def __len__(self):
        if self._len is None:
            self._len = len(self.layer)
        return self._len


class NetworkGeneratar:
    def __init__(self, graph_generator, starts, ends, max_samples, dryrun_args):
        self.graph_generator = graph_generator
        self.graph_generator.draw()
        self.graph = graph_generator.graph
        self.starts = starts
        self.ends = ends
        self.dryrun_args = dryrun_args
        self.modulegen = ModuleGen()
        #self.modulegen.register('conv2d', out_channels=32, kernel_size=1, stride=1)
        self.modulegen.register('linear', out_channels=64)
        self.modulegen.register('identity')
        self.interface = NetworkxInterface(self.graph)
        self.max_samples = max_samples
        self.subgraph = self.get_subgraph(starts, ends, max_samples)
        self.n_subgraph = len(self.subgraph)
        self._len = None

    def get_subgraph(self, starts, ends, max_samples):
        return self.interface.sample(starts, ends, max_samples)

    def _construct_module(self, edge_list, _idx):
        module = Graph()
        for i in self.starts:
            vec = self.modulegen.get_identity_vec()
            module.add_input_node(f'{i}', vec=vec)
        node_dict = {}

        for (src, dst) in [list(self.graph.edges())[i-1] for i in edge_list]:
            src = int(src)
            dst = int(dst)
            if not dst in node_dict.keys():
                node_dict[dst] = [src]
            else:
                node_dict[dst].append(src)
        print('self.graph.edges()', self.graph.edges())
        print('self.graph.nodes()', self.graph.nodes())                        
        print('node_dict', node_dict)
        for key, previous in sorted(node_dict.items(), key=lambda x: x[0]):
            layer_idx = _idx % len(self.modulegen)
            _idx //= len(self.modulegen)
            layer_names = [m._get_name() for m in module.modules()][1:] # skip 'Graph' module
            node_names = list(module.graph.nodes)
            print('node_names', node_names)
            if len(previous) == 1:
                mod, vec = self.modulegen[layer_idx]
                if mod._get_name() == 'Conv2d':
                    parents_indexes = [node_names.index(str(p)) for p in previous]
                    parents_module_names = [layer_names[i] for i in parents_indexes]
                    if 'FlattenLinear' in parents_module_names:
                        raise RuntimeError("can't append Conv2d after FlattenLinear.")
                module.add_node(f'{key}', mod, previous=[str(p) for p in previous], vec=vec)
            else:
                mod, vec = self.modulegen.get_cat()
                parents_indexes = [node_names.index(str(p)) for p in previous]
                print('parents_indexes', previous)                                
                parents_module_names = [layer_names[i] for i in parents_indexes]
                if 'FlattenLinear' in parents_module_names and 'Conv2d' in parents_module_names:
                    raise RuntimeError("can't concatinate FlattenLinear and Conv2d")
                module.add_node(f'{key}', mod, previous=[str(p) for p in previous], vec=vec)

        mod, vec = self.modulegen.get_linear(10, None)
        module.add_node(f'{int(key) + 1}', mod, vec=vec, previous=[f'{key}'])
        vec = self.modulegen.get_identity_vec()
        module.add_output_node(f'{int(key) + 2}', f'{int(key) + 1}', vec=vec)
        edges = [[int(e[0]) - 1, int(e[1]) - 1] for e in module.graph.edges()]
        node_features = self.modulegen.get_empty_mat(int(key) + 2)
        for node in module.graph.nodes(data=True):
            idx = int(node[0]) - 1
            node_features[idx, :] = node[1]['vec']
        y = module(*self.dryrun_args)
        return module, (edges, np.vstack(node_features))

    def __iter__(self):
        self.counter = 0
        return self

    def elongate(self, template_graph):
        self._len = None
        template_graph = self.edge_index_to_graph(template_graph, sorted(self.graph.edges()))
        print('template_graph', template_graph)
        original_graph = self.graph_generator.to_original_graph(template_graph)
        print('elongate', original_graph.edges())
        self.graph_generator = self.graph_generator.elongate(original_graph)
        print('self.graph_generator.graph.edges()', self.graph_generator.graph.edges())
        self.graph_generator.draw(filename='elongated_graph.png')
        self.graph = self.graph_generator.renamed_graph()
        print('self.graph.edges()', self.graph.edges())
        self.interface = NetworkxInterface(self.graph)
        
        _end_nodes = list(sorted(self.graph.nodes()))[len(self.graph.nodes()) - self.graph_generator.n_outputs:]
        print('start node', self.graph_generator.start_nodes)
        print('end node', _end_nodes)
        self.subgraph = self.get_subgraph(self.graph_generator.start_nodes,
                                          _end_nodes,
                                          self.max_samples)
        self.n_subgraph = len(self.subgraph)
        # ng = NetworkGeneratar(self.graph_generator,
        #                       self.graph_generator.start_nodes,
        #                       self.graph_generator.end_nodes,
        #                       self.max_samples,
        #                       dryrun_args=self.dryrun_args)
        return self

    def draw(self, idx, filename='test.png'):
        edge_index_list = self.subgraph[idx]
        graph = self.edge_index_to_graph(edge_index_list, sorted(self.graph.edges()))
        self.graph_generator.draw(graph, filename)

    def edge_index_to_graph(self, edge_index_list, edges):
        g = nx.DiGraph()
        edges = self.edge_index_to_edges(edge_index_list, edges)
        for e in edges:
            g.add_edge(*e)
        return g
    
    def edge_index_to_edges(self, edge_index_list, edges):
        return [list(edges)[int(edge_index)-1] for edge_index in edge_index_list]

    def __next__(self):
        if self.counter <= len(self):
            self.counter += 1
            while True:
                try:
                    module, edges, node_features = self[self.counter]
                    module(*self.dryrun_args)
                    break
                except RuntimeError as e:
                    self.counter += 1
                    pass
            return module, edges, node_features
        else:
            raise StopIteration

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError
        _idx = idx
        subgraph_idx = _idx % self.n_subgraph
        edge_list = self.subgraph[subgraph_idx]
        _idx //= self.n_subgraph
        module = self._construct_module(edge_list, _idx)
        return module

    def __len__(self):
        if self._len is None:
            n_layer = len(self.modulegen)
            n = 0
            for graph in self.subgraph:
                n_edges = len(graph) # return number of edges
                n += n_layer ** n_edges
            self._len = n
        return self._len


def objectve(trial):
    model, _ = trial.graph
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
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
    s = time.time()
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
        if batch_idx > 10:
            break
    acc = 100. * correct / ((batch_idx + 1) * batch_size)
    print(acc)
    return acc


    
if __name__ == "__main__":
    generator = BlockGenerator(2, 1)
    sample_size = 5
    ns = NetworkxInterface(generator.graph)
    graphs = ns.sample(generator.start_nodes, generator.end_nodes, 100)
    x = torch.rand(128, 392)
    # 392 + 392 # for linear layer
    ng = NetworkGeneratar(generator, generator.start_nodes, generator.end_nodes, 300, dryrun_args=(x, x))
    models = []
    num_node_features = 4
    best_acc = 0

    # for i in range(ng.n_subgraph):
    #     ng.draw(i, f'g{i:03}.png')
    for i in range(10):
        searcher = Searcher()
        print('size of ng', len(ng))
        if len(ng) == 0:
            ng.graph_generator.draw(filename='size0.png')
        samples = np.random.randint(0, len(ng), size=sample_size)        
        searcher.register_trial('graph', [ng[i] for i in samples])
        n_trials = 2
        n_random_trials = 1
        model_kwargs = dict(
            input_dim=num_node_features,
            n_train_epochs=400,
        )
        # result = searcher.search(objectve,
        #                          n_trials=n_trials,
        #                          deep_surrogate_model=f'thdbonas.deep_surrogate_models:GCNSurrogateModel',
        #                          n_random_trials=n_random_trials,
        #                          model_kwargs=model_kwargs)
        # print(f'{i} trial', result.max_value_idx, result.best_trial, result.best_value)
        # if result.best_value > best_acc:
        #     best_acc = result.best_value
        # else:
        #     print(result.best_trial)
        
        # print(ng[result.max_value_idx][0].graph)
        # print('best_graph-{i}.png', ng[result.max_value_idx][0].graph.edges())
        warnings.warn(f'{ng[100][0].graph.edges()}')


        print(ng[100][0].graph)
        ng.graph_generator.draw(ng[100][0].graph, f'best_graph-{i}.png')
        
        # ng = ng.elongate(ng[result.max_value_idx][0].graph)
        # print('elongated_graph-{i}.png', ng.graph.edges())
        #ng.graph_generator.draw(ng.graph, f'elongated_graph-{i}.png')
        # for j in range(ng.n_subgraph):
        #     ng.draw(j, f'g{i:03}-{j:03}.png')

