import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_graph import Graph
import copy

class Embed(nn.Module):
    def __init__(self, graph, alphas, embed_dim=64, num_iterations=4):
        super(Embed, self).__init__()
        self.graph = graph
        self.num_iterations = num_iterations
        self.embed_dim = embed_dim

        self.alpha1, self.alpha2, self.alpha3, self.alpha4 = alphas

        # Initialize node embeddings as zero vectors on the correct device.
        self.cur_embed = torch.zeros(self.graph.num_nodes, self.embed_dim).to(self.alpha1.device)
    
    def re_init(self):
        self.graph.labels = [0] * self.graph.num_nodes
        self.cur_embed = torch.zeros(self.graph.num_nodes, self.embed_dim).to(self.alpha1.device)

    def copy_emb(self):
        new_emb = Embed(copy.deepcopy(self.graph), (self.alpha1, self.alpha2, self.alpha3, self.alpha4),
                        self.embed_dim, self.num_iterations)
        new_emb.cur_embed = self.cur_embed.clone().detach()
        return new_emb
     
    def update(self):
        num_nodes = self.graph.num_nodes
        labels = torch.tensor(self.graph.labels, dtype=torch.float32).to(self.alpha1.device)
        x = self.cur_embed  

        for _ in range(self.num_iterations):
            new_x = torch.zeros_like(x)
            
            for v in range(num_nodes):
                # Create neighbor_sum on the same device as x.
                neighbor_sum = torch.zeros(self.embed_dim, device=x.device)
                # Initialize edge_sum as a tensor.
                edge_sum = torch.zeros(1, device=x.device)
        
                for (u, weight) in self.graph.adj[v]:
                    neighbor_sum += x[u]
                    edge_sum += F.relu(self.alpha3 * weight)

                new_x[v] = F.relu(self.alpha1 * neighbor_sum + self.alpha2 * edge_sum + self.alpha4 * labels[v])
            
            x = new_x

        self.cur_embed = x
