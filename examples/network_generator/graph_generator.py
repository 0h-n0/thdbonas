import networkx as nx
from networkx.algorithms.components import strongly_connected_components

import itertools
from typing import List

from test_data_generator import make_graph
from layer import conv_output_size

import random
from functools import reduce
from operator import and_
from frame_generator import FrameGenerator


def make_size_transition_graph(max_size: int, kernel_sizes: List[int], strides: List[int]):
    g = nx.DiGraph()
    for x_in in range(1, max_size + 1):
        for s, k in itertools.product(kernel_sizes, strides):
            x_out = conv_output_size(x_in, k, s)
            if x_out > 0: g.add_edge(x_in, x_out)
    return g


class GraphGenerator():
    """
    与えられるグラフについて、各nodeに有効な出力サイズを割り振ります。
    """

    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        max_size: int,
        allow_param_in_concat: bool,  # concat + conv などを許すか
        kernel_sizes: List[int],
        strides: List[int]
    ):
        self.g = g
        self.n_nodes = max(g.nodes)
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.allow_param_in_concat = allow_param_in_concat
        self.size_transision_graph = make_size_transition_graph(max_size, kernel_sizes, strides)
        self.scc_idx, self.g_compressed = self.compress_graph()
        self.g_compressed_inv = self.g_compressed.reverse()
        self.t_sorted = list(nx.topological_sort(self.g_compressed))

    def compress_graph(self):
        """ 出力サイズが同じ頂点を縮約したグラフを作る """
        g_for_scc = self.g.copy()
        for v in self.g.nodes:
            # vの入力があるnodeとvのoutput_sizeは同じ
            if self.is_concat_node(v):
                if self.allow_param_in_concat:
                    v_edges = list(self.g_inv.edges([v]))
                    for (_, f), (__, t) in zip(v_edges[:-1], v_edges[1:]):
                        g_for_scc.add_edge(f, t)
                        g_for_scc.add_edge(t, f)
                else:
                    for _, u in self.g_inv.edges([v]):
                        g_for_scc.add_edge(v, u)

        scc_idx = [0] * (self.n_nodes + 1)
        scc = strongly_connected_components(g_for_scc)
        for idx, nodes in enumerate(scc):
            for v in nodes:
                scc_idx[v] = idx

        g_compressed = nx.DiGraph()
        for v in self.g.nodes:
            g_compressed.add_node(scc_idx[v])
        for s, t in self.g.edges:
            rs = scc_idx[s]
            rt = scc_idx[t]
            if rs != rt: g_compressed.add_edge(rs, rt)

        return scc_idx, g_compressed

    def is_concat_node(self, v: int) -> bool:
        return len(self.g_inv.edges([v])) >= 2

    def dfs(self, v_idx: int, g_labeled: nx.DiGraph, valid_size_lists: List[List[int]]):
        v = self.t_sorted[v_idx]
        is_end = v == self.t_sorted[-1]
        # どこでもいいのでvに入る頂点をpick up
        s = list(self.g_compressed_inv.edges([v]))[0][1]
        # 割り当て可能なsizeを探す
        for _, sz in self.size_transision_graph.edges([g_labeled.nodes[s]['size']]):
            validities = [
                self.size_transision_graph.has_edge(g_labeled.nodes[u]['size'], sz) for _, u in self.g_compressed_inv.edges([v])
            ]
            is_valid_size = reduce(and_, validities)
            if is_valid_size:
                g_labeled.nodes[v]['size'] = sz
                if is_end:
                    valid_size_lists.append([g_labeled.nodes[v]['size'] for v in self.t_sorted])
                else:
                    self.dfs(v_idx + 1, g_labeled, valid_size_lists)

    def as_size_dict(self, size_list):
        """ self.t_sortedに対応するsizeを{vertex: size}のdictに変換する """
        size_dict = {}
        for v in self.g.nodes:
            rv = self.scc_idx[v]
            idx = self.t_sorted.index(rv)
            size_dict[v] = size_list[idx]
        return size_dict

    # TODO 動かなくなっているので直す
    def list_valid_output_sizes(self, input_size: int):
        if len(self.t_sorted) == 1:
            return [self.as_size_dict([input_size])]
        g_labeled = nx.DiGraph()
        start = self.t_sorted[0]
        g_labeled.add_nodes_from(self.t_sorted)
        g_labeled.nodes[start]['size'] = input_size
        ans = []
        self.dfs(1, g_labeled, ans)
        return [self.as_size_dict(l) for l in ans]

    def sample_valid_output_size(self, input_sizes: List[int], max_failures=100):
        """
        有効な出力サイズを探して１つ返します。
        max_failures回失敗したら諦めてFalseを返します
        """
        assert len(input_sizes) == len(self.starts)
        find = False
        starts = [self.scc_idx[s] for s in self.starts]
        fail_count = 0
        while not find:
            g_labeled = nx.DiGraph()
            g_labeled.add_nodes_from(self.t_sorted)
            for v, s in zip(starts, input_sizes): g_labeled.nodes[v]['size'] = s

            for v in self.t_sorted:
                if v in starts: continue
                is_end = v == self.t_sorted[-1]
                s = list(self.g_compressed_inv.edges([v]))[0][1]  # どこでもいいのでvに入る頂点をpick up
                valid_sizes = []  # 割り当て可能なsize
                for _, sz in self.size_transision_graph.edges([g_labeled.nodes[s]['size']]):
                    validities = [
                        self.size_transision_graph.has_edge(g_labeled.nodes[u]['size'], sz) for _, u in self.g_compressed_inv.edges([v])
                    ]
                    is_valid_size = reduce(and_, validities)
                    if is_valid_size:
                        valid_sizes.append(sz)
                if len(valid_sizes) == 0:
                    fail_count += 1
                    # print(f"failed on {v}")
                    if fail_count >= max_failures: return False
                    break

                g_labeled.nodes[v]['size'] = random.sample(valid_sizes, 1)[0]
                # print(f"{v}: {g_labeled.nodes[v]['size']}", end="")
                if is_end:
                    return self.as_size_dict([g_labeled.nodes[v]['size'] for v in self.t_sorted])
