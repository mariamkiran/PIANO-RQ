import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from collections import deque

from custom_graph import Graph
from stovec import Embed
from gymenv import CustomEnv
from simulator import simulate
from simulator import celf


#hyperparameters
REPLAY_CAPACITY = 2000
GAMMA = 0.99
LR = 0.001
EPSILON = 0.1



class QNet(nn.Module):
    def __init__(self, embed_dim=32):
        super(QNet, self).__init__()
        self.embed_dim = embed_dim

        # Trainable parameters beta_1, beta_2, beta_3
        self.beta1 = nn.Parameter(torch.rand(embed_dim, 1))  # Shape: [embed_dim, 1]
        self.beta2 = nn.Parameter(torch.rand(1))            # Scalar
        self.beta3 = nn.Parameter(torch.rand(1))            # Scalar

        # Final linear transformation to produce scalar Q value
        self.fc = nn.Linear(embed_dim * 2, 1)

    def forward(self, node_embed, agg_embed):
        
        scaled_aggregate = self.beta2 * agg_embed
        scaled_node = self.beta3 * node_embed

        # Concatenate scaled embeddings
        combined = torch.cat((scaled_aggregate, scaled_node), dim=1)  # Shape: [1, 2 * embed_dim]

        # Apply ReLU and final transformation
        q_value = self.fc(F.relu(combined))  # Shape: [1, 1]
        return q_value


class DQNAgent:

    def __init__(self, embed_dim=32):
        """
        Initializes the DQN agent.
        Args:
            embed: The embeddings_
            graph: The graph object containing adjacency and labels.
            embed_dim: Dimensionality of node embeddings.
        """
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
        self.embed_dim = embed_dim
        
        # OPTIMIZER: trains both alphas and betas
        self.q_network = QNet(embed_dim)
        
        self.alpha1 = nn.Parameter(torch.rand(1))
        self.alpha2 = nn.Parameter(torch.rand(1))
        self.alpha3 = nn.Parameter(torch.rand(1))
        self.alpha4 = nn.Parameter(torch.rand(1))
        self.shared_alphas = [self.alpha1, self.alpha2, self.alpha3, self.alpha4]

        # Optimizer trains both QNet and shared betas
        self.optimizer = optim.Adam(
            list(self.q_network.parameters()) + self.shared_alphas, 
            lr=LR
        )

    def select_action(self, env, valid_nodes, epsilon):
        """
        Args:
            valid_nodes: List of node indices that are not yet in the seed set.
        Returns:
            The index of the selected node.
        """

        # Compute the current embeddings
        current_embeddings = env.embed.cur_embed
        agg_embed = current_embeddings.sum(dim=0)  # Sum of all node embeddings

        
        if random.random() < epsilon:
            # Exploration: Choose a random valid node
            return random.choice(valid_nodes)
        else:
            # Exploitation: Choose the node with the highest Q-value
            q_values = []
            for v in valid_nodes:
                node_embed = current_embeddings[v]  # Embedding for node v
                q_values.append(
                    self.q_network(node_embed.unsqueeze(0), agg_embed.unsqueeze(0))
                )
            q_values = torch.cat(q_values).squeeze()
            return valid_nodes[q_values.argmax().item()] #basically return which node has largest Q value

    def train(self, batch_size):
        """
        Trains the agent using experiences from the replay buffer.

        Args:
            batch_size: Number of samples to use for training.
        """
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        loss = 0

        for state, action, reward, next_state in batch:

            agg_embed = state.cur_embed.sum(dim=0)  # Aggregate embedding for state
            node_embed = state.cur_embed[action]  # Embedding of the selected action node

            # Compute Q(s, a) for the current state-action pair
            q_value = self.q_network(node_embed.unsqueeze(0), agg_embed.unsqueeze(0))

            # Compute embeddings for the next state

            next_agg_embed = next_state.cur_embed.sum(dim=0)

            # Compute the target Q-value using max_a' Q(s', a')
            max_next_q = max(
                self.q_network(next_state.cur_embed[v].unsqueeze(0), next_agg_embed.unsqueeze(0))
                for v in range(state.graph.num_nodes) if next_state.graph.labels[v] != 1
            )

            target = reward + GAMMA * max_next_q

            # Compute loss for this experience
            loss += (target - q_value)

        #delayed loss update
        loss = loss ** 2
        loss /= batch_size
        self.optimizer.zero_grad() 
        loss.backward()  # Backpropagate to update neural network parameters
        self.optimizer.step()  # Update alpha and beta

    def add_experience(self, state, action, reward, next_state):
        state_copy = state.copy_emb()
        next_state_copy = next_state.copy_emb()
        self.replay_buffer.append((state_copy, action, reward, next_state_copy))

    def evaluate(self, env, budget):
        start_time = time.time()
        env.embed.update()

        agg_embed = env.embed.cur_embed.sum(dim=0) 
        #NODE NUMBER MUST START WITH ZERO!
        q_list = []
        for v in range(env.embed.graph.num_nodes):
            node_embed = env.embed.cur_embed[v]
            q_value = self.q_network(node_embed.unsqueeze(0), agg_embed.unsqueeze(0)).item()
            q_list.append((q_value,v))
        q_list.sort(reverse=True)

        for (q,v) in q_list:
            if budget<=0:
                break
            env.embed.graph.labels[v] = 1
            budget-=1
        
        result = simulate(env.embed.graph,10000)

        end_time = time.time()

        print(end_time-start_time)

        env.reset()

        
        return result
    
    def random_select(self, env, nodes, num_nodes):
        env.embed.update()
        random_numbers = random.sample(range(num_nodes), nodes)
        for i in random_numbers:
            env.embed.graph.labels[i] = 1
        result = simulate(env.embed.graph,1000)
        env.reset()
        return result
    

        


def train_agent(agent, env, episodes, batch_size):
    """
    Trains the DQN agent by interacting with the CustomEnv.

    Args:
        agent: Instance of DQNAgent.
        env: Instance of CustomEnv.
        episodes: Number of episodes to train.
        batch_size: Batch size for training.
    """

    for episode in range(episodes):
        # Reset the environment to get the initial embeddings
        state = env.reset()  

        done = False
        episode_reward = 0

        epsilon = 0.10

        while not done:
            # Get the valid nodes (not yet selected)
            valid_nodes = [i for i, label in enumerate(env.embed.graph.labels) if label == 0]

            
            #for i in range(env.embed.graph.num_nodes):
            #   if (env.embed.graph.labels[i]!=1):
            #        valid_nodes.append(i)

            # Select an action 
            action = agent.select_action(env, valid_nodes, epsilon)
            epsilon *= 0.95

            # Apply the action 
            next_state, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer
            agent.add_experience(state, action, reward, next_state)

            # Train the agent
            agent.train(batch_size)

            # Update the state
            state = next_state
            episode_reward += reward

            
        # Log episode performance
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {episode_reward}")
        print(f"total influenced: {env.influence}")

    agent.replay_buffer.clear()
    torch.save({
        'q_network_state_dict': agent.q_network.state_dict(),
        'shared_alphas_state_dict': {f'alpha{i+1}': alpha for i, alpha in enumerate(agent.shared_alphas)}
    }, 'C:\\Users\\17789\\Desktop\\Graph Dataset\\DQN_agent.pth')

    
def DQN_main(num_nodes):
    
    
    input_file = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\subgraph1.txt'
    adj_list = {}

    for i in range(num_nodes+1): adj_list[i]= []
    max_node = 0
    with open(input_file, 'r') as file:
        for line in file:
            u, v, weight = line.strip().split()  # Split into u, v, and weight
            u, v = int(u), int(v)  # Convert node IDs to integers
            weight = float(weight)  # Convert weight to float
            
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append((v, weight))  # Add edge with weight
            max_node = max(max_node, max(u,v))

    
    if max_node<100 :
        return
    
    max_node = max_node
    # Create a Graph object with the adjacency list
    graph = Graph(max_node+1, adj_list)
    

    agent = DQNAgent()

    if os.path.exists('C:\\Users\\17789\\Desktop\\Graph Dataset\\DQN_agent.pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('C:\\Users\\17789\\Desktop\\Graph Dataset\\DQN_agent.pth')
        agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    
        # Restore shared alphas
        shared_alphas_state_dict = checkpoint['shared_alphas_state_dict']
        for i, alpha in enumerate(agent.shared_alphas):
            alpha.data = shared_alphas_state_dict[f'alpha{i+1}']
    else:
        print("No pre-trained agent found. Creating a new agent...")
    env = CustomEnv(graph, agent.shared_alphas, 10)

   
    train_agent(agent, env, 30, 10)    

    '''
    random_avg = 0.0
    for i in range(20):
        random_avg += agent.random_select(env, 10,max_node)

    print(f'random result: {random_avg/20}')
    
    print(agent.evaluate(env, 10))
    print(celf(graph,10))
    '''



