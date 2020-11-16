import networkx as nx
from networkx.algorithms.components import strongly_connected_components

import itertools
from typing import List, Dict
import random
from functools import reduce
from operator import and_, or_

from .layer import conv_output_size
from .frame_generator import FrameGenerator


def make_size_transition_graph(max_size: int, kernel_sizes: List[int], strides: List[int]):
    """ 遷移可能なサイズ間に有向辺を張ったグラフを作成します """
    g = nx.DiGraph()
    for x_in in range(1, max_size + 1):
        g.add_edge(x_in, x_in)  # NOTE: identityのために自己辺は必ず張っておく
        for k, s in itertools.product(kernel_sizes, strides):
            x_out = conv_output_size(x_in, k, s)
            if x_out > 0: g.add_edge(x_in, x_out)
    return g


class OutputSizeSearcher():
    """
    与えられるグラフについて、各nodeに有効な出力サイズを割り振るためのクラス
    Attributes
    -------
    g_compressed: gの形状から同じ出力サイズになることが要請される頂点を縮約したグラフ  
    scc_idx: scc_idx[v]で、gの頂点vに対するg_compressedの頂点の番号を取得できる  
    t_sorted: g_compressedの頂点をトポロジカル順序に並べたリスト   
    allow_param_in_concat: concatの頂点でconvolutionなどの別処理をすることを許すか否か
    """

    def __init__(
        self,
        g: nx.DiGraph,
        starts: List[int],
        ends: List[int],
        max_input_size: int,
        allow_param_in_concat: bool,
        kernel_sizes: List[int],
        strides: List[int]
    ):
        self.g = g
        self.max_node_idx = max(g.nodes)
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends
        self.allow_param_in_concat = allow_param_in_concat
        self.size_transision_graph = make_size_transition_graph(max_input_size, kernel_sizes, strides)
        self.scc_idx, self.g_compressed = self.compress_graph()
        self.g_compressed_inv = self.g_compressed.reverse()
        self.t_sorted = list(nx.topological_sort(self.g_compressed))

    def compress_graph(self):
        """ 出力サイズが同じ頂点を縮約したグラフを作る """
        g_for_scc = self.g.copy()
        for v in self.g.nodes:
            # vからの入力があるnodeとvのoutput_sizeは同じ
            if self.is_concat_node(v):
                if self.allow_param_in_concat:
                    v_edges = list(self.g_inv.edges([v]))
                    for (_, s), (__, t) in zip(v_edges[:-1], v_edges[1:]):
                        g_for_scc.add_edge(s, t)
                        g_for_scc.add_edge(t, s)
                else:
                    for _, u in self.g_inv.edges([v]):
                        g_for_scc.add_edge(v, u)

        scc_idx = [0] * (self.max_node_idx + 1)
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

    def __as_size_dict(self, size_list):
        """ self.t_sortedに対応するsizeを{vertex: size}のdictに変換する """
        size_dict = {}
        for v in self.g.nodes:
            rv = self.scc_idx[v]
            idx = self.t_sorted.index(rv)
            size_dict[v] = size_list[idx]
        return size_dict

    # topological順序で見るのでこれていい
    def __list_reachable_nodes_in_compressed_graph(self, v: int):
        """ 縮約されたグラフ上でvから到達可能な点を全て列挙します """
        reachabilities = [False] * len(self.t_sorted)
        reachable_nodes = [v]
        reachabilities[v] = True
        for u in self.t_sorted:
            if u == v or len(self.g_compressed_inv.edges([u])) == 0:
                continue
            is_reachable = reduce(or_, [reachabilities[s] for (_, s) in self.g_compressed_inv.edges([u])])
            if is_reachable:
                reachabilities[u] = True
                reachable_nodes.append(u)
        return reachable_nodes

    def sample_output_dimensions(self, n_seed_nodes=3):
        """
        有効な出力の次元を１つ返します。  
        １次元になる頂点をn_seed_nodes個決め、それらより下は１次元にするという感じの動作をします。
        Returns
        ----------
        nodeの番号がkey, 出力の次元(1 or 4)がvalueのdict  
        """
        middle_nodes = set(self.g.nodes) - (set(self.starts) - set(self.ends))
        middle_node_scc_indices = list({self.scc_idx[v] for v in middle_nodes})
        seed_nodes = random.sample(middle_node_scc_indices, n_seed_nodes)
        one_dimensional_nodes = reduce(
            or_, [set(self.__list_reachable_nodes_in_compressed_graph(seed_node)) for seed_node in seed_nodes]
        )
        return {v: 1 if self.scc_idx[v] in one_dimensional_nodes else 4 for v in self.g.nodes}

    def __list_valid_output_sizes_of_node(self, g_labeled: nx.DiGraph, v: int):
        nodes = [u for _, u in self.g_compressed_inv.edges([v])]  # vに入る頂点たち
        return list(reduce(and_, [{sz for _, sz in self.size_transision_graph.edges([g_labeled.nodes[u]['size']])} for u in nodes]))

    def sample_valid_output_size(self, input_sizes: Dict[int, int], output_dimensions: Dict[int, int], max_failures=100):
        """
        有効な出力サイズを探して１つ返します。
        Parameters
        ----------
        input_sizes: input nodeの番号がkey, 入力サイズがvalueのdict  
        output_dimensions: nodeの番号がkey, 出力の次元(1 or 4)がvalueのdict   
        max_failures: max_failures回失敗したら諦めてFalseを返します
        Returns
        ----------
        output_sizes: nodeの番号がkey, 出力のサイズがvalueのdict(次元が1のnodeについては本関数では決めず-1を返す) 
        """
        assert len(input_sizes) == len(self.starts)
        scc_idx_output_dimensions = {self.scc_idx[v]: output_dimensions[v] for v in self.g.nodes}
        starts = [self.scc_idx[s] for s in self.starts]
        fail_count = 0
        while fail_count < max_failures:
            g_labeled = nx.DiGraph()
            g_labeled.add_nodes_from(self.t_sorted)

            # 入力の頂点はそのままのサイズで出力する
            for v, sz in input_sizes.items():
                g_labeled.nodes[self.scc_idx[v]]['size'] = sz

            for v in self.t_sorted:
                if v in starts: continue
                # 出力が1次元のものはここでは決めない
                if scc_idx_output_dimensions[v] == 1:
                    g_labeled.nodes[v]['size'] = -1
                else:
                    valid_sizes = self.__list_valid_output_sizes_of_node(g_labeled, v)
                    if len(valid_sizes) == 0:
                        fail_count += 1
                        break
                    g_labeled.nodes[v]['size'] = random.choice(valid_sizes)

                v_is_end = v == self.t_sorted[-1]
                if v_is_end:
                    return self.__as_size_dict([g_labeled.nodes[v]['size'] for v in self.t_sorted])

        return False
