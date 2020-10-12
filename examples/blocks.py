#!/usr/bin/env python
import networkx as nx


class BlockGenerator:
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.graph = nx.DiGraph()
        self._n_elongation = 1
        self.graph = self._init(self.graph, n_inputs, n_outputs)

    def _init(self, graph, n_inputs, n_outputs):
        counter = 1
        for i in range(n_inputs):
            graph.add_edge(counter, counter + n_inputs)
            counter += 1
        graph, counter = self._construct_itermidiate_layers(graph, n_inputs, n_inputs + 1)
        graph, counter = self._construct_output_layers(graph, n_inputs, n_outputs, counter)                     
        return graph
    
    def _construct_itermidiate_layers(self, graph, n_inputs, counter):
        for _ in range(n_inputs):
            if counter in self.graph.nodes():            
                graph.add_edge(counter, counter + n_inputs)
                graph.add_edge(counter, counter + n_inputs + 1)
                graph.add_edge(counter + n_inputs + 1, counter + 2 * n_inputs + 1)
            graph.add_edge(counter + n_inputs, counter + 2 * n_inputs + 1)                                    
            graph.add_edge(counter, counter + 2 * n_inputs + 1)                        
            counter += 1
        return graph, counter

    def _construct_output_layers(self, graph, n_inputs, n_outputs, counter):
        counter = counter + n_inputs + 1                                    
        for i in range(n_inputs, n_outputs, -1):
            for j in range(0, i-1):
                graph.add_edge(counter, counter + i)
                graph.add_edge(counter + 1, counter + i)
                graph.add_edge(counter - i, counter + i)
                counter += 1
            counter += 1
        for i in range(n_outputs):
            graph.add_edge(counter+i, counter + n_outputs + i)
        return graph, counter
    
    def elongate(self, template_graph: nx.Graph):
        self.graph = self._construct_from_template(template_graph)
        self._n_elongation += 1
        return self

    def _construct_from_template(self, template_graph):
        ## remove nodes
        n_removed_layer = self.n_inputs - self.n_outputs
        n_remained_nodes = len(template_graph.nodes) - self.n_outputs
        for i in range(n_removed_layer):
            n_remained_nodes -= self.n_outputs + i
        removed_nodes = list(template_graph.nodes)[n_remained_nodes:]
        template_graph.remove_nodes_from(removed_nodes)
        graph = template_graph
        
        ## construct graph
        n_inputs = self.n_inputs
        n_outputs = self.n_outputs        
        counter = self.n_inputs + (self.n_inputs + self.n_inputs + 1) + 1
        graph, counter = self._construct_itermidiate_layers(graph, n_inputs, counter)
        graph, _ = self._construct_output_layers(graph, n_inputs, n_outputs, counter)
        return graph
    
    def draw(self, filename='graph.png'):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        pos_dir = self._create_position(self.n_inputs, self.n_outputs)
        labels = {n: n for n in self.graph.nodes()}                            
        nx.draw(self.graph, pos_dir, ax=ax, labels=labels)
        plt.savefig(filename)
        plt.clf()
        
    def _create_position(self, n_inputs, n_outputs):
        pos_dir = {}
        counter = 1
        depth = 0
        ## input layers
        for i in range(n_inputs):
            pos_dir[counter] = (i, depth)
            counter += 1
        depth -= 0.5
        for i in range(n_inputs):
            pos_dir[counter] = (i, depth)
            counter += 1
        ## intermidiate layers
        for _ in range(self._n_elongation):
            depth -= 0.5
            for j in range(n_inputs + 1):            
                pos_dir[counter] = (j-0.5, depth)
                counter += 1
            depth -= 0.5                
            for j in range(n_inputs):            
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
                pos_dir[counter] = (x_pos+j, depth)
                counter += 1
        ## final layer
        depth -= 0.5
        if self.n_outputs > i:
            x_pos = self.n_inputs//2 - self.n_outputs/2 + diff
        else:
            x_pos = self.n_inputs//2 - i/2 + diff
        for i in range(n_outputs):
            pos_dir[counter] = (x_pos+i, depth)            
            counter += 1
        return pos_dir
        
    
if __name__ == '__main__':
    b = BlockGenerator(5, 2)
    b.draw('graph1.png')
    b.graph.remove_node(16)
    b.graph.remove_node(14)
    b.graph.remove_node(18)        
    b.elongate(b.graph)
    b.draw()
