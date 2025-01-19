import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_graph import Graph
import copy
#torch.set_default_device('cuda')

class Embed(nn.Module):
    def __init__(self, graph, alphas, embed_dim=32, num_iterations=4):
        
        #embed_dim is q in the DISCO article. Currently arbitrary
        #graph contains number of nodes, a 2d adj vector, and a label list for seed state
        #set num_iterations to 4 as DISCO said it should usually be 4 or less.

        super(Embed, self).__init__()
        self.graph = graph
        self.num_iterations = num_iterations
        self.embed_dim = embed_dim

        self.alpha1, self.alpha2, self.alpha3, self.alpha4 = alphas

        # Initialize node embeddings as zero vectors
        self.cur_embed = torch.zeros(self.graph.num_nodes, self.embed_dim).to(self.alpha1.device)
    
    def re_init(self):
        self.graph.labels = [0] * self.graph.num_nodes
        self.cur_embed = torch.zeros(self.graph.num_nodes, self.embed_dim).to(self.alpha1.device)

    def copy_emb(self):
        new_emb = Embed(copy.deepcopy(self.graph), (self.alpha1, self.alpha2, self.alpha3, self.alpha4))
        new_emb.cur_embed = self.cur_embed.clone().detach()
        return new_emb
     
    def update(self):
        """
        update the embeddings based on the graph structure.
        """
        num_nodes = self.graph.num_nodes
        labels = torch.tensor(self.graph.labels, dtype=torch.float32).to(self.alpha1.device)
        x = self.cur_embed  

        # Iteratively update embeddings
        for _ in range(self.num_iterations):
            new_x = torch.zeros_like(x)  # Temporary tensor to store updated embeddings
            
            # Update embeddings for each node based on neighbors embedding in last iter
            for v in range(num_nodes):
                neighbor_sum = torch.zeros(self.embed_dim)
                edge_sum = 0.0

        
                for (u, weight) in self.graph.adj[v]:
                    neighbor_sum += x[u]  
                    edge_sum += F.relu(self.alpha3 * weight)  

                # Update the embedding of node v using formula in DISCO paper
                new_x[v] = F.relu(
                    self.alpha1 * neighbor_sum +
                    self.alpha2 * edge_sum +
                    self.alpha4 * labels[v]
                )
            
            x = new_x  # Update embeddings

        self.cur_embed = x
