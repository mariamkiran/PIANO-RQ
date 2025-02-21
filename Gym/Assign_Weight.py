import random
from collections import deque

def bfs_sample(input_file, output_file, start_node, max_nodes):
    # Read the input adjacency list
    adj_list = []
    with open(input_file, 'r') as file:
        for line in file:
            u, v = map(int, line.strip().split())
            weight = round(random.uniform(0.25, 0.75), 2)
            adj_list.append((u,v,weight))

    
    with open(output_file, 'w') as file:
        for u, v, w in adj_list:
            file.write(f"{u} {v} {w}\n")

bfs_sample('C:\\Users\\17789\\Desktop\\New Graph Dataset\\p2p(2).txt', 'C:\\Users\\17789\\Desktop\\New Graph Dataset\\p2p(2).txt', 0, 10000)
