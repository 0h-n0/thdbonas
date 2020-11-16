import networkx as nx
import numpy as np
import copy
from typing import List, Dict
from collections import OrderedDict
import matplotlib.pyplot as plt

from inferno.extensions.containers import Graph
from inferno.extensions.layers.reshape import Concatenate

import torchex.nn as exnn
import torch.nn as nn

from .layer import find_conv_layer, conv2d, ConcatConv, FlattenLinear, ConcatFlatten 
from .tox21_layers import (Tox21GCNConv, Tox21GlobalMeanPool, Tox21Linear, Tox21ReLU, Tox21Embedding,
                           Tox21FlattenLinear, Tox21ConcatFlatten, Tox21Concat, Tox21ConcatLinear,
                           Tox21GCNorLinear)

class ModuleVec:
    def __init__(self):
        self.layer_dict = OrderedDict()
        self._num_features = 0

    def register(self, layer_name: str):
        self.layer_dict[layer_name] = 0.0
        self._num_features += 1

    def get_vector(self, param_dict: Dict[str, int]):
        _layer_dict = copy.deepcopy(self.layer_dict)
        for k, v in param_dict.items():
            _layer_dict[k] = v
        vec = list(_layer_dict.values())
        return vec

    @property
    def num_features(self):
        return self._num_features

# constructorがまあまあなロジック持ってるけどどうしよう
class NNModuleGenerator():
    """
    Attributes(分かりにくそうなもののみ記載):
       network_input_sizes: (各入力のnodeについて)nodeの番号がkey, 入力サイズがvalueのdict 
       node_output_sizes: (各nodeについて)nodeの番号がkey, 出力サイズがvalueのdict 
       network_output_sizes: (各出力のnodeについて)nodeの番号がkey, 出力サイズがvalueのdict 
    """

    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        network_input_sizes: Dict[int, int],
        node_output_sizes: Dict[int, int],
        node_output_dimensions: Dict[int, int],
        network_output_sizes: Dict[int, int],
        kernel_sizes: List[int],
        strides: List[int],
        output_channel_candidates: List[int],
        n_nodes: int,
    ):
        self.g = g
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.network_input_sizes = network_input_sizes
        self.node_output_sizes = node_output_sizes
        self.node_output_dimensions = node_output_dimensions
        self.network_output_sizes = network_output_sizes
        self.node_input_sizes = self.get_input_sizes()
        self.node_input_dimensions = self.get_input_dimensions()
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.output_channels = self.__calc_output_channels(output_channel_candidates)
        self.module_vec = ModuleVec()
        [self.module_vec.register(l) for l in ['identity', 'concat', 'conv2d', 'linear', 'flatten',
                                               'out_channels', 'kernel', 'stride', 'relu']]
        self.n_nodes = n_nodes
        self.n_features = self.module_vec.num_features

    # TODO この辺りは順番次第では動かないので直す
    def is_concat_flatten_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_dimensions[v] != self.node_input_dimensions[v]

    def is_flatten_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) <= 1 and self.node_output_dimensions[v] != self.node_input_dimensions[v]

    def is_concat_conv_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_sizes[v] != self.node_input_sizes[v]

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_sizes[v] == self.node_input_sizes[v]

    def is_linear_node(self, v: int) -> bool:
        return self.node_output_dimensions[v] == 1 and self.node_input_dimensions[v] == 1

    def get_input_sizes(self):
        input_sizes = {}
        for v, s in self.network_input_sizes.items():
            input_sizes[v] = s
        for s, t in self.g.edges:
            input_sizes[t] = self.node_output_sizes[s]
        return input_sizes

    def get_input_dimensions(self):
        input_dimensions = {}
        for v in self.starts:
            input_dimensions[v] = 4
        for s, t in self.g.edges:
            input_dimensions[t] = self.node_output_dimensions[s]
        return input_dimensions

    def add_layer(self, v: int, module):
        previous_nodes = [f"{u}" for (_, u) in self.g_inv.edges([v])]
        out_channels = self.output_channels[v]
        node_feature = None
        if v in self.starts:
            module.add_input_node(f"{v}")
            node_feature_dict = dict(
                identity=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)
        elif v in self.ends:
            module.add_node(f"{v}", previous=previous_nodes, module=FlattenLinear(out_channels))
            node_feature_dict = dict(
                flatten=1, linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)
        elif self.is_concat_flatten_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=ConcatFlatten(out_channels))
            node_feature_dict = dict(
                flatten=1, concat=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_flatten_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=FlattenLinear(out_channels))
            node_feature_dict = dict(
                concat=1, linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_concat_conv_node(v):
            k, s = find_conv_layer(self.node_input_sizes[v], self.node_output_sizes[v], self.kernel_sizes, self.strides)
            module.add_node(
                f"{v}", previous=previous_nodes, module=ConcatConv(out_channels=out_channels, kernel_size=k, stride=s))
            node_feature_dict = dict(
                concat=1, conv2d=1, kernel=k, stride=s, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_concat_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Concatenate())
            node_feature_dict = dict(
                concat=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_linear_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=exnn.Linear(out_channels))
            node_feature_dict = dict(
                linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.node_output_sizes[v] == self.node_input_sizes[v]:
            module.add_node(f"{v}", previous=previous_nodes, module=nn.ReLU())
            node_feature_dict = dict(
                relu=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        else:
            k, s = find_conv_layer(self.node_input_sizes[v], self.node_output_sizes[v], self.kernel_sizes, self.strides)
            module.add_node(
                f"{v}", previous=previous_nodes, module=conv2d(out_channels=out_channels, kernel_size=k, stride=s))
            node_feature_dict = dict(
                conv2d=1, kernel=k, stride=s, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        return node_feature

    def __calc_output_channels(self, output_channel_candidates: List[int]):
        """
        親のoutput_channelsの和以下のものがcandidatesにあったらその内最大のものを採用。
        そうでないときはmin(candidates)を採用
        """
        output_channels = {v: 3 for v in self.starts}
        self.g_inv = self.g.reverse()
        for v in sorted(list(self.g.nodes)):
            if v in self.starts: continue
            sum_inputs = sum([output_channels[u] for (_, u) in self.g_inv.edges(v)])
            if sum_inputs < min(output_channel_candidates):
                output_channels[v] = min(output_channel_candidates)
            else:
                output_channels[v] = max(filter(lambda x: x <= sum_inputs, output_channel_candidates))
        return output_channels

    def run(self):
        module = Graph()
        node_features = np.zeros((self.n_nodes, self.n_features))
        for v in sorted(list(self.g.nodes)):
            node_feature = self.add_layer(v, module)
            node_features[v] = np.array(node_feature)
        module.add_node('concat', previous=[f"{t}" for t in self.ends], module=Concatenate())
        # node_feature_dict = dict(concat=1)
        # node_features.append(self.module_vec.get_vector(node_feature_dict))
        module.add_output_node('output', previous='concat')
        # node_feature_dict = dict(identity=1)
        # node_features.append(self.module_vec.get_vector(node_feature_dict))
        return module, node_features



# constructorがまあまあなロジック持ってるけどどうしよう
class ChemblModuleGenerator():
    """
    Attributes(分かりにくそうなもののみ記載):
       network_input_sizes: (各入力のnodeについて)nodeの番号がkey, 入力サイズがvalueのdict 
       node_output_sizes: (各nodeについて)nodeの番号がkey, 出力サイズがvalueのdict 
       network_output_sizes: (各出力のnodeについて)nodeの番号がkey, 出力サイズがvalueのdict 
    """

    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        network_input_sizes: Dict[int, int],
        node_output_sizes: Dict[int, int],
        node_output_dimensions: Dict[int, int],
        network_output_sizes: Dict[int, int],
        output_channel_candidates: List[int],
        n_nodes: int,
        batch_size: int
    ):
        self.g = g
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.network_input_sizes = network_input_sizes
        self.node_output_sizes = node_output_sizes
        self.node_output_dimensions = node_output_dimensions
        self.batch_size = batch_size
        self.network_output_sizes = network_output_sizes
        self.node_input_sizes = self.get_input_sizes()
        self.node_input_dimensions = self.get_input_dimensions()
        self.output_channels = self.__calc_output_channels(output_channel_candidates)
        self.module_vec = ModuleVec()
        [self.module_vec.register(l) for l in ['identity', 'concat', 'gcn', 'linear', 'flatten',
                                               'out_channels', 'relu', 'embd', 'pool']]
        self.n_nodes = n_nodes
        self.n_features = self.module_vec.num_features

    # TODO この辺りは順番次第では動かないので直す
    def is_concat_flatten_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_dimensions[v] != self.node_input_dimensions[v]

    def is_flatten_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) <= 1 and self.node_output_dimensions[v] != self.node_input_dimensions[v]

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_sizes[v] == self.node_input_sizes[v]

    def is_linear_node(self, v: int) -> bool:
        return self.node_output_dimensions[v] == 1 and self.node_input_dimensions[v] == 1

    def get_input_sizes(self):
        input_sizes = {}
        for v, s in self.network_input_sizes.items():
            input_sizes[v] = s
        for s, t in self.g.edges:
            input_sizes[t] = self.node_output_sizes[s]
        return input_sizes

    def get_input_dimensions(self):
        input_dimensions = {}
        for v in self.starts:
            input_dimensions[v] = 4
        for s, t in self.g.edges:
            input_dimensions[t] = self.node_output_dimensions[s]
        return input_dimensions

    def add_layer(self, v: int, module):
        previous_nodes = [f"{u}" for (_, u) in self.g_inv.edges([v])]
        out_channels = self.output_channels[v]
        node_feature = None
        if v in self.starts:
            module.add_input_node(f"{v}")
            node_feature_dict = dict(
                identity=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)
        elif v == 2:
            # GCN
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21GCNConv(out_channels=out_channels))
            node_feature_dict = dict(
                gcn=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif v == 3:
            # emmbeding
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21Embedding())
            node_feature_dict = dict(
                flatten=1, embd=1, pool=1, out_channels=64)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif v in self.ends:
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21FlattenLinear(out_channels, self.batch_size))
            node_feature_dict = dict(
                flatten=1, linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)
        elif self.is_concat_flatten_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21ConcatFlatten(out_channels, self.batch_size))
            node_feature_dict = dict(
                flatten=1, concat=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_flatten_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21FlattenLinear(out_channels, self.batch_size))
            node_feature_dict = dict(
                concat=1, linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_concat_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21Concat(self.batch_size))
            node_feature_dict = dict(
                concat=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_linear_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21GCNorLinear(out_channels, self.batch_size))
            node_feature_dict = dict(
                linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.node_output_sizes[v] == self.node_input_sizes[v]:
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21ReLU())
            node_feature_dict = dict(
                relu=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        else:
            r = np.random.randint(0, 2)
            if r == 0:
                module.add_node(
                    f"{v}", previous=previous_nodes, module=Tox21GCNConv(out_channels=out_channels))
                node_feature_dict = dict(
                    gcn=1, out_channels=out_channels)
                node_feature = self.module_vec.get_vector(node_feature_dict)
            else:
                module.add_node(
                    f"{v}", previous=previous_nodes, module=Tox21GlobalMeanPool())
                node_feature_dict = dict(
                    pool=1)
                node_feature = self.module_vec.get_vector(node_feature_dict)
        return node_feature

    def __calc_output_channels(self, output_channel_candidates: List[int]):
        """
        親のoutput_channelsの和以下のものがcandidatesにあったらその内最大のものを採用。
        そうでないときはmin(candidates)を採用
        """
        output_channels = {v: 3 for v in self.starts}
        self.g_inv = self.g.reverse()
        for v in sorted(list(self.g.nodes)):
            if v in self.starts: continue
            sum_inputs = sum([output_channels[u] for (_, u) in self.g_inv.edges(v)])
            if sum_inputs < min(output_channel_candidates):
                output_channels[v] = min(output_channel_candidates)
            else:
                output_channels[v] = max(filter(lambda x: x <= sum_inputs, output_channel_candidates))
        return output_channels

    def run(self, out_channels):
        module = Graph()
        node_features = np.zeros((self.n_nodes, self.n_features))
        for v in sorted(list(self.g.nodes)):
            node_feature = self.add_layer(v, module)
            node_features[v] = np.array(node_feature)
        module.add_node('concat', previous=[f"{t}" for t in self.ends], module=Tox21ConcatLinear(out_channels))
        # node_feature_dict = dict(concat=1)
        # node_features.append(self.module_vec.get_vector(node_feature_dict))
        module.add_output_node('output', previous='concat')
        # node_feature_dict = dict(identity=1)
        # node_features.append(self.module_vec.get_vector(node_feature_dict))
        return module, node_features


class Tox21ModuleGenerator():
    """
    Attributes(分かりにくそうなもののみ記載):
       network_input_sizes: (各入力のnodeについて)nodeの番号がkey, 入力サイズがvalueのdict 
       node_output_sizes: (各nodeについて)nodeの番号がkey, 出力サイズがvalueのdict 
       network_output_sizes: (各出力のnodeについて)nodeの番号がkey, 出力サイズがvalueのdict 
    """

    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        network_input_sizes: Dict[int, int],
        node_output_sizes: Dict[int, int],
        node_output_dimensions: Dict[int, int],
        network_output_sizes: Dict[int, int],
        output_channel_candidates: List[int],
        n_nodes: int,
        batch_size: int,
    ):
        self.g = g
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.network_input_sizes = network_input_sizes
        self.node_output_sizes = node_output_sizes
        self.node_output_dimensions = node_output_dimensions
        self.network_output_sizes = network_output_sizes
        self.batch_size = batch_size
        self.node_input_sizes = self.get_input_sizes()
        self.node_input_dimensions = self.get_input_dimensions()
        self.output_channels = self.__calc_output_channels(output_channel_candidates)
        self.module_vec = ModuleVec()
        [self.module_vec.register(l) for l in ['identity', 'concat', 'gcn', 'linear', 'flatten',
                                               'out_channels', 'relu', 'pool']]
        self.n_nodes = n_nodes
        self.n_features = self.module_vec.num_features

    # TODO この辺りは順番次第では動かないので直す
    def is_concat_flatten_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_dimensions[v] != self.node_input_dimensions[v]

    def is_flatten_node(self, v) -> bool:
        return len(self.g_inv.edges([v])) <= 1 and self.node_output_dimensions[v] != self.node_input_dimensions[v]

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2 and self.node_output_sizes[v] == self.node_input_sizes[v]

    def is_linear_node(self, v: int) -> bool:
        return self.node_output_dimensions[v] == 1 and self.node_input_dimensions[v] == 1

    def get_input_sizes(self):
        input_sizes = {}
        for v, s in self.network_input_sizes.items():
            input_sizes[v] = s
        for s, t in self.g.edges:
            input_sizes[t] = self.node_output_sizes[s]
        return input_sizes

    def get_input_dimensions(self):
        input_dimensions = {}
        for v in self.starts:
            input_dimensions[v] = 4
        for s, t in self.g.edges:
            input_dimensions[t] = self.node_output_dimensions[s]
        return input_dimensions

    def add_layer(self, v: int, module):
        previous_nodes = [f"{u}" for (_, u) in self.g_inv.edges([v])]
        out_channels = self.output_channels[v]
        node_feature = None
        if v in self.starts:
            module.add_input_node(f"{v}")
            node_feature_dict = dict(
                identity=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)
        elif v == 1:
            # GCN
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21GCNConv(out_channels=out_channels))
            node_feature_dict = dict(
                gcn=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif v in self.ends:
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21FlattenLinear(out_channels, self.batch_size))
            node_feature_dict = dict(
                flatten=1, linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)
        elif self.is_concat_flatten_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21ConcatFlatten(out_channels, self.batch_size))
            node_feature_dict = dict(
                flatten=1, concat=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_flatten_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21FlattenLinear(out_channels, self.batch_size))
            node_feature_dict = dict(
                concat=1, linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_concat_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21Concat(self.batch_size))
            node_feature_dict = dict(
                concat=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.is_linear_node(v):
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21GCNorLinear(out_channels, self.batch_size))
            node_feature_dict = dict(
                linear=1, out_channels=out_channels)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        elif self.node_output_sizes[v] == self.node_input_sizes[v]:
            module.add_node(f"{v}", previous=previous_nodes, module=Tox21ReLU())
            node_feature_dict = dict(
                relu=1)
            node_feature = self.module_vec.get_vector(node_feature_dict)            
        else:
            r = np.random.randint(0, 2)
            if r == 0:
                module.add_node(
                    f"{v}", previous=previous_nodes, module=Tox21GCNConv(out_channels=out_channels))
                node_feature_dict = dict(
                    gcn=1, out_channels=out_channels)
                node_feature = self.module_vec.get_vector(node_feature_dict)
            else:
                module.add_node(
                    f"{v}", previous=previous_nodes, module=Tox21GlobalMeanPool())
                node_feature_dict = dict(
                    pool=1)
                node_feature = self.module_vec.get_vector(node_feature_dict)
        return node_feature

    def __calc_output_channels(self, output_channel_candidates: List[int]):
        """
        親のoutput_channelsの和以下のものがcandidatesにあったらその内最大のものを採用。
        そうでないときはmin(candidates)を採用
        """
        output_channels = {v: 3 for v in self.starts}
        self.g_inv = self.g.reverse()
        for v in sorted(list(self.g.nodes)):
            if v in self.starts: continue
            sum_inputs = sum([output_channels[u] for (_, u) in self.g_inv.edges(v)])
            if sum_inputs < min(output_channel_candidates):
                output_channels[v] = min(output_channel_candidates)
            else:
                output_channels[v] = max(filter(lambda x: x <= sum_inputs, output_channel_candidates))
        return output_channels

    def run(self, out_channels):
        module = Graph()
        node_features = np.zeros((self.n_nodes, self.n_features))
        for v in sorted(list(self.g.nodes)):
            node_feature = self.add_layer(v, module)
            node_features[v] = np.array(node_feature)

        for t in self.ends:
            module.add_node(f't{t}', previous=[f"{t}",], module=Tox21Linear(out_channels, self.batch_size, False))
            module.add_output_node('output', previous=f't{t}')            
        # node_feature_dict = dict(concat=1)
        # node_features.append(self.module_vec.get_vector(node_feature_dict))
        # node_feature_dict = dict(identity=1)
        # node_features.append(self.module_vec.get_vector(node_feature_dict))
        return module, node_features
    

if __name__ == '__main__':
    pass
                          
