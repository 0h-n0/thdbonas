#!/usr/bin/env python
import logging

import networkx as nx
from frontier_graph import NetworkxInterface


def get_logger(level='DEBUG'):
    FORMAT = '%(asctime)-15s - %(pathname)s - %(funcName)s - L%(lineno)3d ::: %(message)s'
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger


class BlockGenerator:
    def __init__(self, n_inputs, n_outputs, logger=None):
        self.inv_flag = n_inputs < n_outputs
        if self.inv_flag:
            self.n_inputs = n_outputs
            self.n_outputs = n_inputs
        else:
            self.n_inputs = n_inputs
            self.n_outputs = n_outputs
        if logger is None:
            self.logger = get_logger()
        self._graph = nx.DiGraph()
        self._n_elongation = 1
        self._graph = self._init(self._graph, self.n_inputs, self.n_outputs)
        _nodes = sorted(self._graph.nodes())
        self._start_nodes = list(_nodes)[:n_inputs]
        self._end_nodes = list(_nodes)[len(self._graph.nodes()) - n_outputs:]
        self.logger.debug(f'self._end_nodes = {self._end_nodes}')
        
    @property
    def start_nodes(self):
        return self._start_nodes            
        
    @property
    def end_nodes(self):
        return self._end_nodes            
        
    @property
    def graph(self):
        if self.inv_flag:
            return self._inverse_graph(self._graph)
        else:
            return self._graph
            
    def _inverse_graph(self, graph):
        inv_graph = nx.DiGraph()
        nodes = [i for i in sorted(graph.nodes())]
        nodes.reverse()
        to_inv_nodes_dict = {i: v for i, v in enumerate(nodes)}
        for e in graph.edges():
            inv_node0 = to_inv_nodes_dict[e[0]]
            inv_node1 = to_inv_nodes_dict[e[1]]
            inv_graph.add_edge(inv_node1, inv_node0)
        return inv_graph

    def _init(self, graph, n_inputs, n_outputs):
        counter = 0
        for i in range(n_inputs):
            graph.add_edge(counter, counter + n_inputs)
            counter += 1
        graph, counter = self._construct_intermidiate_layers(graph, n_inputs)
        graph, _ = self._construct_output_layers(graph, n_inputs, n_outputs, counter)
        return graph
    
    def _construct_intermidiate_layers(self, graph, n_inputs):
        self.logger.debug(f'self._n_elongation = {self._n_elongation}')
        counter = n_inputs + 1 + (self._n_elongation - 1) * (n_inputs) - 1
        if self._n_elongation > 2:            
            counter += (n_inputs - 1) * (self._n_elongation - 2)
        self.logger.debug(f'counter = {counter}')        
        # print('-------->', (self._n_elongation - 2) * (2 * n_inputs - 1))
        # print('counter', counter)
        # print(self._graph.nodes())
        if self._n_elongation > 1:
            # create direct arrow
            #     1
            #  2     3
            #     4
            #  5     6
            #  2 -> 5:
            #  3 -> 6:
            for _ in range(n_inputs - 1):
                if counter in graph.nodes():
                    bottom_arrow = [counter, counter + 2 * (n_inputs - 1) + 1]
                    # print('direct arrow', _counter, _counter + 2 * n_inputs + 1)
                    self.logger.debug(f'bottom_arrow {bottom_arrow}')
                    #graph.add_edge(*bottom_arrow)
                    counter += 1
                    
        self.logger.debug(f'diamond_base: counter = {counter}')
        left_side_nodes = [n for n in range(n_inputs,
                                            1 + n_inputs + self._n_elongation * (n_inputs + n_inputs -1),
                                            n_inputs + n_inputs -1)]
        right_side_nodes = [n for n in range(2 * n_inputs - 1,
                                             2 * n_inputs + self._n_elongation * (n_inputs + n_inputs -1),
                                             n_inputs + n_inputs -1)]
        self.logger.debug(f'left_side_nodes = {left_side_nodes}')
        self.logger.debug(f'right_side_nodes = {right_side_nodes}')        
        for _ in range(n_inputs):
            # create diamond graph
            #    1
            #  2   3
            #    4
            # 1 -> 2: to left arrow
            # 1 -> 3: to right arrow
            # 1 -> 4: bottom arrow
            # 2 -> 4: from left arrow
            # 3 -> 4: from right arrow
            if counter in graph.nodes():
                to_left_arrow = [counter, counter + n_inputs - 1]
                from_left_arrow = [counter + n_inputs - 1, counter + 2 * n_inputs - 1]
                to_right_arrow = [counter, counter + n_inputs]
                from_right_arrow = [counter + n_inputs, counter + 2 * n_inputs - 1]
                bottom_arrow = [counter, counter + 2 * (n_inputs - 1) + 1]

                if not counter in left_side_nodes:
                    graph.add_edge(*to_left_arrow)
                    graph.add_edge(*from_left_arrow)
                    self.logger.debug(f'to left arrow   , {to_left_arrow[0]:3}, {to_left_arrow[1]:3}')
                    self.logger.debug(f'from left arrow , {from_left_arrow[0]:3}, {from_left_arrow[1]:3}')
                    
                if not counter in right_side_nodes:
                    graph.add_edge(*to_right_arrow)
                    graph.add_edge(*from_right_arrow)
                    self.logger.debug(f'to right arrow  , {to_right_arrow[0]:3}, {to_right_arrow[1]:3}')
                    self.logger.debug(f'from right arrow, {from_right_arrow[0]:3}, {from_right_arrow[1]:3}')
                graph.add_edge(*bottom_arrow)
                self.logger.debug(f'bottom arrow    , {bottom_arrow[0]:3}, {bottom_arrow[1]:3}')
            counter += 1
        return graph, counter + self.n_inputs - 1

    def _construct_output_layers(self, graph, n_inputs, n_outputs, counter):
        self.logger.debug(f'counter = {counter}')
        for i in range(n_inputs, n_outputs, -1):
            for j in range(0, i-1):
                to_right_arrow = [counter, counter + i]
                to_left_arrow = [counter + 1, counter + i]
                if i == n_inputs:
                    bottom_arrow = [counter - i + 1, counter + i]
                else:
                    bottom_arrow = [counter - i, counter + i]                    
                self.logger.debug(f'to left arrow   , {to_left_arrow[0]:3}, {to_left_arrow[1]:3}')
                self.logger.debug(f'to right arrow  , {to_right_arrow[0]:3}, {to_right_arrow[1]:3}')
                self.logger.debug(f'bottom arrow    , {bottom_arrow[0]:3}, {bottom_arrow[1]:3}')
                if i == n_inputs:
                    if to_left_arrow[0] in graph.nodes():
                        graph.add_edge(*to_left_arrow)
                    if to_right_arrow[0] in graph.nodes():                        
                        graph.add_edge(*to_right_arrow)
                else:
                    graph.add_edge(*to_left_arrow)
                    graph.add_edge(*to_right_arrow)                    
                #graph.add_edge(*bottom_arrow)
                counter += 1
            counter += 1
        for i in range(n_outputs):
            graph.add_edge(counter+i, counter + n_outputs + i)
        return graph, counter
    
    def elongate(self, template_graph: nx.Graph=None):
        if template_graph is None:
            template_graph = self._graph
        self._n_elongation += 1
        self.logger.debug(f"template_graph.edges: {template_graph.edges}")
        self.logger.debug(f"self._n_elongation : {self._n_elongation}")        
        self._graph = self._construct_from_template(template_graph)
        self._end_nodes = list(sorted(self._graph.nodes()))[len(self._graph.nodes()) - self.n_outputs:]
        self.logger.debug(f"updated self._end_nodes : {self._end_nodes}")
        return self

    def _construct_from_template(self, template_graph):
        n_removed_layer = self.n_inputs - self.n_outputs
        self.logger.debug(f"n_removed_layer : {n_removed_layer}")        
        n_removed_nodes = 1
        for i in range(n_removed_layer):
            n_removed_nodes += self.n_outputs + i
        self.logger.debug(f"n_removed_nodes : {n_removed_nodes}")
            
        nodes = list(sorted(template_graph.nodes))
        #self.logger.debug(f'remained_nodes {nodes[:n_remained_nodes]}')        
        biggest_node = self.n_inputs * 2 + (2 * self.n_inputs - 1) * (self._n_elongation - 1)
        for i in range(self.n_inputs):
            biggest_node += i
        biggest_node += 1
        self.logger.debug(f"biggest_node : {biggest_node}")        
        removed_nodes_list = [i for i in range(biggest_node - n_removed_nodes, biggest_node)]
        self.logger.debug(f'removed_nodes_list : {removed_nodes_list}')
        template_graph.remove_nodes_from(removed_nodes_list)
        graph = template_graph
        ## construct graph
        graph, counter = self._construct_intermidiate_layers(graph, self.n_inputs)
        graph, _ = self._construct_output_layers(graph, self.n_inputs, self.n_outputs, counter)
        return graph
    
    def draw(self, graph=None, filename='graph.png', string_node=False):
        if graph is None:
            self.logger.debug(f'self.inv_flag {self.inv_flag}')
            if self.inv_flag:
                graph = self._graph.reverse()
            else:
                graph = self._graph
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos_dir = self._create_position(graph, self.n_inputs, self.n_outputs, string_node)
        labels = {n: n for n in graph.nodes()}
        nx.draw(graph, pos_dir, ax=ax, labels=labels)
        plt.savefig(filename)
        plt.clf()
        
    def _create_position(self, graph, n_inputs, n_outputs, string_node):
        pos_dir = {}
        for i in range(len(graph.nodes()) + 5) :
            if string_node:
                pos_dir[str(i)] = (0, 0.1)
            else:
                pos_dir[i] = (0, 0.1)
        counter = 0
        depth = 0
        ## input layers
        for i in range(n_inputs):
            if string_node:
                pos_dir[str(counter)] = (i, depth)
            else:
                pos_dir[counter] = (i, depth)                
            counter += 1
        depth -= 0.5
        for i in range(n_inputs):
            if string_node:
                pos_dir[str(counter)] = (i, depth)
            else:
                pos_dir[counter] = (i, depth)                
            counter += 1
        ## intermidiate layers
        for _ in range(self._n_elongation):
            depth -= 0.5
            for j in range(n_inputs - 1):
                if string_node:
                    pos_dir[str(counter)] = (j+0.5, depth)                    
                else:
                    pos_dir[counter] = (j+0.5, depth)                

                counter += 1
            depth -= 0.5                
            for j in range(n_inputs):
                if string_node:
                    pos_dir[str(counter)] = (j, depth)
                else:
                    pos_dir[counter] = (j, depth)
                counter += 1

        if self.n_inputs % 2 != 0:
            diff = 0.5
        else:
            diff = 0
        ## output layers
        for i in range(self.n_inputs-1, n_outputs-1, -1):
            depth -= 0.5
            x_pos = self.n_inputs // 2 - i / 2 + diff
            for j in range(i):
                if string_node:
                    pos_dir[str(counter)] = (x_pos+j, depth)
                else:
                    pos_dir[counter] = (x_pos+j, depth)                    
                counter += 1
        ## final layer
        depth -= 0.5
        if self.n_outputs > i:
            x_pos = self.n_inputs//2 - self.n_outputs/2 + diff
        else:
            x_pos = self.n_inputs//2 - i/2 + diff
        for i in range(n_outputs):
            if string_node:
                pos_dir[str(counter)] = (x_pos+i, depth)
            else:
                pos_dir[counter] = (x_pos+i, depth)                
            counter += 1
        return pos_dir
    
    def renamed_graph(self):
        graph = nx.DiGraph()
        _name_relation_original_to_dummy = {}
        for idx, n in enumerate(sorted(self._graph.nodes())):
            new_name = idx + 1
            graph.add_node(new_name)
            _name_relation_original_to_dummy[n] = new_name
        for idx, e in enumerate(sorted(self._graph.edges())):
            n1 = _name_relation_original_to_dummy[e[0]]
            n2 = _name_relation_original_to_dummy[e[1]]            
            graph.add_edge(n1, n2)
        return graph

    def to_original_graph(self, renamed_graph):
        graph = nx.DiGraph()
        _name_relation_original_to_dummy = {}
        for idx, n in enumerate(sorted(self._graph.nodes())):
            new_name = idx + 1
            _name_relation_original_to_dummy[n] = new_name
        _name_relation_dummy_to_original = {i: k for (k, i) in
                                            _name_relation_original_to_dummy.items()}
        for idx, e in enumerate(sorted(renamed_graph.edges())):
            n1 = _name_relation_dummy_to_original[e[0]]
            n2 = _name_relation_dummy_to_original[e[1]]            
            graph.add_edge(n1, n2)
        return graph

    
if __name__ == '__main__':

    for n in range(2, 12):
        b = BlockGenerator(n, 1)
        b.draw(filename=f'graph{n}-{0}.png')            
        for e in range(1, 6):
            b.elongate()
            b.draw(filename=f'graph{n}-{e}.png')                        
    # for i in range(3):
    #     b.draw(filename=f'graph{i}.png')
    #     
    # interface = NetworkxInterface(b.graph)
    # subgraph = interface.sample(b.start_nodes, b.end_nodes, sample_size)
    #for idx, s in enumerate(subgraph):
    # idx = 000
    # subgraph = interface.edge_indices_to_digraph(subgraph[-2])
    # b.draw(subgraph, filename=f'subgraph{idx:03d}.png')        
    # print(list(subgraph.degree))                
    # b.elongate(subgraph)
    # b.draw(filename=f'elongated_subgraph{idx:03d}.png')
    # idx = 1    
    # graph = b.renamed_graph()
    # interface = NetworkxInterface(graph)
    # subgraph = interface.sample(b.start_nodes, [list(graph.nodes)[-1],], sample_size)
    # subgraph = interface.edge_indices_to_digraph(subgraph[-3])    
    # b.draw(b.to_original_graph(subgraph), filename=f'subgraph{idx:03d}.png')            
    # b.elongate(b.to_original_graph(subgraph))
    # b.draw(filename=f'elongated_subgraph{idx:03d}.png')    
    # idx = 2
    # graph = b.renamed_graph()
    # interface = NetworkxInterface(graph)
    # subgraph = interface.sample(b.start_nodes, [list(graph.nodes)[-1],], sample_size)
    # subgraph = interface.edge_indices_to_digraph(subgraph[-3])    
    # b.draw(b.to_original_graph(subgraph), filename=f'subgraph{idx:03d}.png')            
    # b.elongate(b.to_original_graph(subgraph))
    # b.draw(filename=f'elongated_subgraph{idx:03d}.png')    

    
