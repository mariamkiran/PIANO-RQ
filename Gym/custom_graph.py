
class Graph:
    def __init__(self, num_nodes, adj):
        #num_nodes. Number of nodes, assuming nodes numbered from 1 to n
        #adj. Adjacency matrix. if tuple (u,w) in the vth row, edge (v,u) exists and has a weight w
        #label = 1 if node is in seed set
        self.num_nodes = num_nodes
        self.adj = adj
        self.labels = [0] *  self.num_nodes