import networkx as nx
import random

from typing import List
import matplotlib.pyplot as plt


def make_graph():
    g = nx.DiGraph()
    starts = [1, 2]
    ends = [9, ]
    g.add_edge(1, 3)
    g.add_edge(2, 4)
    g.add_edge(3, 5)
    g.add_edge(3, 6)
    g.add_edge(4, 5)
    g.add_edge(4, 7)
    g.add_edge(5, 6)
    g.add_edge(5, 7)
    g.add_edge(5, 8)
    g.add_edge(6, 8)
    g.add_edge(7, 8)
    g.add_edge(8, 9)
    return g, starts, ends


def generate_random_graph(size: int, starts: List[int], ends: List[int], p: float) -> nx.DiGraph:
    """
    確率pくらいで辺を張ることでグラフを作成します。
    """
    g = nx.DiGraph()
    g_inv = nx.DiGraph()
    for v in range(1, size):
        if not v in starts and len(g_inv.edges([v])) == 0:
            s = random.randint(1, v - 1)
            g.add_edge(s, v)
            g_inv.add_edge(v, s)
        added = []
        for to in range(v + 1, size + 1):
            if random.random() < p:
                g.add_edge(v, to)
                g_inv.add_edge(to, v)
                added.append(to)
        if len(added) == 0:
            to = random.randint(v + 1, size)
            g.add_edge(v, to)
            g_inv.add_edge(to, v)
    return g


def generate_graph(n_inputs: int, n_outputs: int, max_width: int, max_width_count: int):
    """ 実際に入力されるグラフを作成します。
    Parameters
    ----------
    n_inputs : int
        inputのnode数
    n_outputs : int
        outputのnode数
    max_width_count : int
        幅が最大(n_inputs+1)になる層の数
    """
    assert max_width > n_inputs
    assert max_width > n_outputs
    l = [n_inputs] + list(range(n_inputs, max_width - 1)) + [max_width - 1, max_width] * max_width_count +\
        list(reversed(range(n_outputs, max_width))) + [n_outputs]
    g = nx.DiGraph()
    cumsum = 0
    for n_cur_layer, n_next_layer in zip(l[0:-1], l[1:]):
        # 一つ下に辺を張る
        if n_cur_layer < n_next_layer:
            g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer) for j in range(n_cur_layer)])
            g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer + 1) for j in range(n_cur_layer)])
        elif n_cur_layer == n_next_layer:
            g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer) for j in range(n_cur_layer)])
        else:
            g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer) for j in range(n_next_layer)])
            g.add_edges_from([(cumsum + j + 1, cumsum + j + n_cur_layer) for j in range(n_next_layer)])
        # 二つ下に辺を張る
        # if i + 2 < n_layers:
        #     n_after_next_layer = l[i + 2]
        #     if n_cur_layer <= n_after_next_layer:
        #         g.add_edges_from([(cumsum + j, cumsum + j + n_cur_layer + n_next_layer) for j in range(n_cur_layer)])
        #     else:
        #         g.add_edges_from(
        #             [(cumsum + j + n_cur_layer - n_after_next_layer - 1, cumsum + j + n_cur_layer + n_next_layer)
        #              for j in range(n_after_next_layer)]
        #         )
        cumsum += n_cur_layer

    starts = list(range(0, n_inputs))
    ends = list(range(cumsum, cumsum + n_outputs))
    return g, starts, ends


if __name__ == "__main__":
    g, s, t = generate_graph(4, 4, 5, 2)
    print(s, t)
