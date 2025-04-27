from custom_graph import Graph
import random
import heapq
import time
from collections import deque

def simulate(graph, iter):

    total_activated = 0
    # Track activated nodes
    for _ in range(iter):
        activated = graph.labels.copy()
        newly_activated = deque()
        for i in range(len(activated)):
            if(activated[i]!=0):
                newly_activated.append((i,0.0))
    
        while newly_activated:
            node, decay = newly_activated.popleft()
        
            # Try to influence neighbors
            for (neighbor, weight) in graph.adj[node]:
                if activated[neighbor] != 1:
                    # Attempt to activate the neighbor with probability equal to weight of edge
                    # There are only 1 chance!
                    if max(0.05,random.random()) < weight:
                        
                        activated[neighbor] = 1
                        newly_activated.append((neighbor, decay+0.5))

        total_activated += activated.count(1)

    return total_activated/iter

def celf(graph, k):

    def marginal_gain(node):
        graph.labels[node] = 1
        influence = simulate(graph, 100)  

        graph.labels[node] = 0
        return influence
    
    start_time = time.time()

    heap = []
    selected = 0

    for node in range(graph.num_nodes):
        gain = marginal_gain(node)
        heapq.heappush(heap, (-gain, node, 0))  # Store (-gain, node, flag)
        #print(node)
   
    while selected < k:
        _, node, last_update = heapq.heappop(heap)

        if last_update == selected:
            graph.labels[node] = 1
            selected+=1
        else:
            gain = marginal_gain(node)
            heapq.heappush(heap, (-gain, node, selected))
        #print(selected)

    end_time = time.time()

    print(end_time-start_time)
    return simulate(graph, 10000)