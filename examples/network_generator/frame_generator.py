import networkx as nx

import copy
import random
from typing import List


# TODO random seedを設定できるように
class FrameGenerator():
    def __init__(self, g: nx.DiGraph, starts: List[int], ends: List[int]):
        self.__check_node_order(g)
        self.g = g
        self.g_inv = g.reverse()
        self.starts = starts
        self.ends = ends

    def __check_node_order(self, g: nx.DiGraph):
        for (s, t) in g.edges:
            assert s < t, "edge should be directed from vertex with smaller index to vertex with larger index"

    def __dfs(self, v: int, cur_graph: nx.DiGraph, cur_graph_inv: nx.DiGraph, valid_graphs: List[nx.DiGraph], max_graphs: int):
        if len(valid_graphs) > max_graphs: return
        # endsに含まれていて入次数が0。
        if v in self.ends and len(cur_graph_inv.edges([v])) == 0:
            return
        # 最後の頂点
        if v == max(self.ends):
            # 使われていない頂点は除いたgraphを作成する
            g_generated = nx.DiGraph()
            g_generated.add_edges_from(cur_graph.edges)
            valid_graphs.append(g_generated)
            return

        # 自分への入次数が0かつstartsに含まれない
        if len(cur_graph_inv.edges([v])) == 0 and (not v in self.starts):
            self.__dfs(v + 1, cur_graph, cur_graph_inv, valid_graphs, max_graphs)
            return

        # 自分への入次数が1以上かstart
        edges = self.g.edges([v])
        # for edge_selection in range(1, 1 << len(edges)):
        for edge_selection in reversed(list(range(1, 1 << len(edges)))):
            for i, (_, to) in enumerate(edges):
                if (1 << i) & edge_selection:
                    cur_graph.add_edge(v, to)
                    cur_graph_inv.add_edge(to, v)
            self.__dfs(v + 1, cur_graph, cur_graph_inv, valid_graphs, max_graphs)
            for i, (_, to) in enumerate(edges):
                if (1 << i) & edge_selection:
                    cur_graph.remove_edge(v, to)
                    cur_graph_inv.remove_edge(to, v)

    def list_valid_graph(self, max_graphs=100):
        start = min(self.starts)
        valid_graphs = []
        self.__dfs(start, nx.DiGraph(), nx.DiGraph(), valid_graphs, max_graphs)
        return valid_graphs

    def sample_graph(self):
        """
        randomに辺をつないでいって１つ部分グラフを返します。全頂点を使用するとは限りません。
        """
        g = nx.DiGraph()
        g_inv = nx.DiGraph()
        # まず上から順に適当に辺を選びながらつないでいく
        for v in sorted(self.g.nodes):
            if len(g_inv.edges([v])) == 0 and not v in self.starts:
                continue
            elif v not in self.ends:
                edges = self.g.edges([v])
                edge_selection = random.randrange(1, 1 << len(edges))
                for i, (_, to) in enumerate(edges):
                    if (1 << i) & edge_selection:
                        g.add_edge(v, to)
                        g_inv.add_edge(to, v)

        # 逆順に見てendsの中でinputからのpathがないものをつないでいく
        for v in reversed(sorted(self.g.nodes)):
            if v not in self.starts and (v in self.ends or len(g.edges([v])) > 0) and len(g_inv.edges([v])) == 0:
                nodes = [s for (_, s) in self.g_inv.edges([v])]
                u = random.choice(nodes)
                g.add_edge(u, v)
                g_inv.add_edge(v, u)
                # edges = self.g_inv.edges([v])
                # edge_selection = random.randrange(1, 1 << len(edges))
                # for i, (_, f) in enumerate(edges):
                #     if (1 << i) & edge_selection:
                #         g.add_edge(f, v)
                #         g_inv.add_edge(v, f)

        return g

    def sample_graph_with_all_nodes(self):
        """
        全頂点を使用するものをsampleします。
        """
        g = nx.DiGraph()
        g_inv = nx.DiGraph()
        # まず上から順に適当に辺を選びながらつないでいく
        for v in self.g.nodes:
            if len(g_inv.edges([v])) == 0 and not v in self.starts:
                nodes = [s for (_, s) in self.g_inv.edges([v])]
                u = random.choice(nodes)
                g.add_edge(u, v)
                g_inv.add_edge(v, u)
            elif v not in self.ends:
                edges = self.g.edges([v])
                edge_selection = random.randrange(1, 1 << len(edges))
                for i, (_, to) in enumerate(edges):
                    if (1 << i) & edge_selection:
                        g.add_edge(v, to)
                        g_inv.add_edge(to, v)
        return g
