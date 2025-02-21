import random
from collections import deque

def bfs_sample(input_file, output_file, start_node, max_nodes):
    # Read the input adjacency list
    adj_list = {}
    with open(input_file, 'r') as file:
        for line in file:
            u, v, w = map(float, line.strip().split())
            if u not in adj_list:
                adj_list[u] = []

            if v not in adj_list:
                adj_list[v] = []

            if v not in adj_list[u]:
                adj_list[u].append((v,w))

            #if u not in adj_list[v]:
                #adj_list[v].append(u)

    # BFS Sampling
    visited = set()
    sampled_nodes = []
    queue = deque([start_node])
    
    while queue and len(sampled_nodes) < max_nodes:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            sampled_nodes.append(node)
            if node in adj_list:  # Add neighbors to the queue
                queue.extend(adj_list[node])

    # Build the subgraph
    subgraph_edges = []
    sampled_set = set(sampled_nodes)
    for node in sampled_nodes:
        if node in adj_list:
            for neighbor in adj_list[node]:
                if neighbor in sampled_set:
                    # Generate a random weight between 0 and 1
                    weight = round(random.uniform(0.25, 0.75), 2)
                    subgraph_edges.append((node, neighbor, weight))
    
    # Re-number nodes starting from 0
    node_mapping = {node: i for i, node in enumerate(sampled_nodes)}
    renumbered_edges = [(node_mapping[u], node_mapping[v], w) for u, v, w in subgraph_edges]

    # Write the output adjacency list
    with open(output_file, 'w') as file:
        for u, v, w in renumbered_edges:
            file.write(f"{u} {v} {w}\n")

#bfs_sample('C:\\Users\\17789\\Desktop\\New Graph Dataset\\p2p(1)', 'C:\\Users\\17789\\Desktop\\New Graph Dataset\\p2p(1)', 0, 10000)
