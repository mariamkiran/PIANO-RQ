from custom_graph import Graph
import random
from collections import deque

def simulate(graph, iter):

    total_activated = 0
    # Track activated nodes
    for _ in range(iter):
        activated = graph.labels.copy()
        newly_activated = deque()
        for i in range(len(activated)):
            if(activated[i]!=0):
                newly_activated.append(i)
    
        while newly_activated:
            node = newly_activated.popleft()
        
            # Try to influence neighbors
            for (neighbor, weight) in graph.adj[node]:
                if activated[neighbor] != 1:
                    # Attempt to activate the neighbor with probability equal to weight of edge
                    # There are only 1 chance!
                    if random.random() < weight:
                        activated[neighbor] = 1
                        newly_activated.append(neighbor)

        total_activated += activated.count(1)

    return total_activated/iter