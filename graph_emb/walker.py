import itertools
import math
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from .alias import create_alias_table, sample_from_alias

def get_neighbors_by_edge_type(graph, node, edge_types):
    """Get neighbors connected by specified edge types."""
    neighbors = []
    for neighbor in graph.neighbors(node):
        edge_data = graph[node][neighbor][0]
        if edge_data.get('EdgeType', 1) in edge_types:
            neighbors.append(neighbor)
    return neighbors

class RandomWalker:
    """Performs random walks on a graph for embedding generation.

    Args:
        graph: NetworkX graph.
        return_param (float): Controls likelihood of revisiting a node (default: 1).
        in_out_param (float): Differentiates inward/outward nodes (default: 1).
        use_rejection_sampling (bool): Whether to use rejection sampling for node2vec (default: True).
    """
    def __init__(self, graph, return_param=1, in_out_param=1, use_rejection_sampling=True):
        self.graph = graph
        self.return_param = return_param
        self.in_out_param = in_out_param
        self.use_rejection_sampling = use_rejection_sampling
        self.start_nodes = [
            node for node in graph.nodes() if graph.nodes[node]['activity'] == 'Logon'
        ]
        print(f"Graph has {len(self.start_nodes)} start nodes (Logon activities)")

    def deepwalk_walk(self, walk_length, start_node, edge_types=[1]):
        """Perform a DeepWalk-style random walk."""
        walk = [start_node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = get_neighbors_by_edge_type(self.graph, current_node, edge_types)
            if not neighbors:
                break
            walk.append(random.choice(neighbors))
        return walk

    def node2vec_walk(self, walk_length, start_node, edge_types):
        """Perform a node2vec walk using alias sampling."""
        walk = [start_node]
        has_higher_edge = False
        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = get_neighbors_by_edge_type(self.graph, current_node, edge_types)
            if not neighbors:
                break
            if len(walk) == 1:
                next_node = neighbors[sample_from_alias(
                    self.alias_nodes[current_node][0],
                    self.alias_nodes[current_node][1]
                )]
            else:
                previous_node = walk[-2]
                edge = (previous_node, current_node)
                next_node = neighbors[sample_from_alias(
                    self.alias_edges[edge][0],
                    self.alias_edges[edge][1]
                )]
                if has_higher_edge or self.graph[previous_node][current_node][0]['EdgeType'] == edge_types[-1]:
                    has_higher_edge = True
            walk.append(next_node)
        return walk, has_higher_edge

    def node2vec_walk_rejection(self, walk_length, start_node, edge_types=[1]):
        """Perform a node2vec walk using rejection sampling."""
        def rejection_sample(return_weight, in_out_weight, num_neighbors):
            upper_bound = max(1.0, max(return_weight, in_out_weight))
            lower_bound = min(1.0, min(return_weight, in_out_weight))
            shatter = 0
            second_upper_bound = max(1.0, in_out_weight)
            if return_weight > second_upper_bound:
                shatter = second_upper_bound / num_neighbors
                upper_bound = second_upper_bound + shatter
            return upper_bound, lower_bound, shatter

        walk = [start_node]
        has_higher_edge = False
        return_weight = 1.0 / self.return_param
        in_out_weight = 1.0 / self.in_out_param

        while len(walk) < walk_length:
            current_node = walk[-1]
            neighbors = get_neighbors_by_edge_type(self.graph, current_node, edge_types)
            if not neighbors:
                break
            if len(walk) == 1:
                next_node = neighbors[sample_from_alias(
                    self.alias_nodes[current_node][0],
                    self.alias_nodes[current_node][1]
                )]
            else:
                upper_bound, lower_bound, shatter = rejection_sample(
                    return_weight, in_out_weight, len(neighbors)
                )
                previous_node = walk[-2]
                previous_neighbors = set(get_neighbors_by_edge_type(self.graph, previous_node, edge_types))
                while True:
                    prob = random.random() * upper_bound
                    if prob + shatter >= upper_bound:
                        next_node = previous_node
                        break
                    next_node = neighbors[sample_from_alias(
                        self.alias_nodes[current_node][0],
                        self.alias_nodes[current_node][1]
                    )]
                    if prob < lower_bound:
                        break
                    if prob < return_weight and next_node == previous_node:
                        break
                    neighbor_prob = 1.0 if next_node in previous_neighbors else in_out_weight
                    if prob < neighbor_prob:
                        break
                if has_higher_edge or self.graph[current_node][next_node][0]['EdgeType'] == edge_types[-1]:
                    has_higher_edge = True
            walk.append(next_node)
        return walk, has_higher_edge

    def simulate_walks(self, edge_types, num_walks, walk_length, min_walk_length, workers=1, verbose=0):
        """Simulate random walks on the graph."""
        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(
                self.start_nodes, num, walk_length, edge_types, min_walk_length
            ) for num in partition_num(num_walks, workers)
        )
        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, edge_types, min_walk_length):
        """Helper function to simulate walks for a subset of nodes."""
        walks = []
        for _ in range(num_walks):
            for node in tqdm(nodes, desc="Simulating walks"):
                if self.return_param == 1 and self.in_out_param == 1:
                    walk = self.deepwalk_walk(walk_length, node, edge_types)
                elif self.use_rejection_sampling:
                    walk, has_higher_edge = self.node2vec_walk_rejection(walk_length, node, edge_types)
                    if len(walk) < min_walk_length or not has_higher_edge:
                        continue
                    walks.append(walk)
                else:
                    walk, has_higher_edge = self.node2vec_walk(walk_length, node, edge_types)
                    if len(walk) < min_walk_length or not has_higher_edge:
                        continue
                    walks.append(walk)
        return walks

    def preprocess_transition_probs(self, edge_types):
        """Precompute transition probabilities for random walks."""
        print(f"Preprocessing transition probabilities for edge types: {edge_types}")
        alias_nodes = {}
        for node in tqdm(self.graph.nodes(), desc="Computing node alias tables"):
            weights = [
                self.graph[node][neighbor][0].get('weight', 1.0)
                for neighbor in get_neighbors_by_edge_type(self.graph, node, edge_types)
            ]
            if not weights:
                alias_nodes[node] = ([1.0], [0])
                continue
            norm_constant = sum(weights)
            normalized_probs = [weight / norm_constant for weight in weights]
            alias_nodes[node] = create_alias_table(normalized_probs)

        if not self.use_rejection_sampling:
            alias_edges = {}
            print("Computing edge alias tables...")
            for edge in self.graph.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1], edge_types)
                if not self.graph.is_directed():
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0], edge_types)
            self.alias_edges = alias_edges

        self.alias_nodes = alias_nodes
        return

    def get_alias_edge(self, source, target, edge_types):
        """Compute transition probabilities for an edge in node2vec."""
        weights = []
        for neighbor in get_neighbors_by_edge_type(self.graph, target, edge_types):
            weight = self.graph[target][neighbor][0].get('weight', 1.0)
            if neighbor == source:
                weights.append(weight / self.return_param)
            elif self.graph.has_edge(neighbor, source):
                weights.append(weight)
            else:
                weights.append(weight / self.in_out_param)
        norm_constant = sum(weights)
        normalized_probs = [weight / norm_constant for weight in weights]
        return create_alias_table(normalized_probs)

class BiasedWalker:
    """Performs biased random walks on a multi-layer graph.

    Args:
        node_to_index (list): Mapping of node IDs to indices.
        temp_path (str): Directory for temporary files.
    """
    def __init__(self, node_to_index, temp_path):
        self.node_to_index = node_to_index
        self.indices = list(range(len(node_to_index)))
        self.temp_path = temp_path

    def simulate_walks(self, num_walks, walk_length, stay_probability=0.3, workers=1, verbose=0):
        """Simulate biased random walks across graph layers."""
        layers_adj = pd.read_pickle(os.path.join(self.temp_path, 'layers_adj.pkl'))
        layers_alias = pd.read_pickle(os.path.join(self.temp_path, 'layers_alias.pkl'))
        layers_accept = pd.read_pickle(os.path.join(self.temp_path, 'layers_accept.pkl'))
        gamma = pd.read_pickle(os.path.join(self.temp_path, 'gamma.pkl'))

        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self._simulate_walks)(
                self.indices, num, walk_length, stay_probability, layers_adj, layers_accept, layers_alias, gamma
            ) for num in partition_num(num_walks, workers)
        )
        walks = list(itertools.chain(*results))
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length, stay_probability, layers_adj, layers_accept, layers_alias, gamma):
        """Helper function to simulate walks for a subset of nodes."""
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self._execute_random_walk(
                    layers_adj, layers_accept, layers_alias, node, walk_length, gamma, stay_probability
                ))
        return walks

    def _execute_random_walk(self, graphs, layers_accept, layers_alias, node, walk_length, gamma, stay_probability=0.3):
        """Execute a single biased random walk."""
        current_layer = 0
        path = [self.node_to_index[node]]

        while len(path) < walk_length:
            if random.random() < stay_probability:
                next_node = choose_neighbor(node, graphs, layers_alias, layers_accept, current_layer)
                path.append(self.node_to_index[next_node])
                node = next_node
            else:
                prob = random.random()
                try:
                    log_value = math.log(gamma[current_layer][node] + math.e)
                    move_up_prob = log_value / (log_value + 1)
                except Exception as e:
                    print(f"Error in layer {current_layer}, node {node}: {e}")
                    raise ValueError("Invalid gamma value")
                if prob > move_up_prob:
                    if current_layer > 0:
                        current_layer -= 1
                else:
                    if (current_layer + 1) in graphs and node in graphs[current_layer + 1]:
                        current_layer += 1
        return path

def choose_neighbor(node, graphs, layers_alias, layers_accept, layer):
    """Choose a neighbor node using alias sampling."""
    neighbors = graphs[layer][node]
    index = sample_from_alias(layers_accept[layer][node], layers_alias[layer][node])
    return neighbors[index]