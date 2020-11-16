import networkx as nx

from typing import List

from .test_data_generator import generate_graph
from .frame_generator import FrameGenerator
from .output_size_searcher import OutputSizeSearcher
from .module_generator import NNModuleGenerator

from torchviz import make_dot
import torch


def list_networks():
    return 0


if __name__ == "__main__":
    kernel_sizes = [1, 2, 3]
    strides = [1, 2, 3]
    output_channel_candidates = [32, 64, 128, 192]
    # edges = [(0, 4), (4, 8), (1, 5), (5, 8), (5, 9), (2, 6), (6, 9), (6, 10), (3, 7), (7, 10), (8, 11), (8, 12), (11, 15), (12, 15), (12, 16), (9, 12), (9, 13), (13, 16), (13, 17), (10, 13), (10, 14), (14, 17), (15, 18), (16, 18), (16, 19), (17, 19), (18, 20), (19, 20), (20, 21)]
    edges = [(11, 14), (8, 11), (8, 10), (10, 13), (7, 10), (7, 9), (9, 12), (6, 8), (5, 8), (5, 7), (3, 6), (3, 5), (2, 5), (2, 4), (4, 7), (1, 3), (1, 2), (0, 1)]
    g = nx.DiGraph()
    for e in edges:
        g.add_edge(*e)
    starts = [0,]
    ends = [12, 13, 14]
    network_input_sizes = {v: 224 for v in starts}
    network_output_sizes = {v: 1 for v in ends}
    
    fg = FrameGenerator(g, starts, ends)
    dryrun_args = tuple([torch.rand(1, 3, s, s) for s in network_input_sizes.values()])

    for idx in range(100):
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
                               kernel_sizes, strides, output_channel_candidates)

        module, _ = mg.run()
        print(oss.g_compressed.edges)
        print(f"found {len(output_sizes)} networks")
        print(opt_sizes)
        print(opt_dims)
        print(f"---example---\noutout sizes:{opt_sizes}\nnetwork:{module}")
        # out = module(*dryrun_args)
        # dot = make_dot(out)
        # dot.format = 'png'
        # dot.render(f'test_outputs/graph_image_{idx}')
