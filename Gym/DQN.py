import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

from custom_graph import Graph
from stovec import Embed
import gymenv

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

    def __init__(self, embed, embed_dim=32):
        """
        Initializes the DQN agent.
        Args:
            embed: The embeddings_
            graph: The graph object containing adjacency and labels.
            embed_dim: Dimensionality of node embeddings.
        """
        self.embed = embed  
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
        self.embed_dim = embed_dim
        
        # OPTIMIZER: trains both alphas and betas
        self.q_network = QNet(embed_dim)
        self.optimizer = optim.Adam(
            list(self.q_network.parameters()) + list(self.embed.parameters()), 
            lr=LR
        )

    def select_action(self, valid_nodes):
        """
        Args:
            valid_nodes: List of node indices that are not yet in the seed set.
        Returns:
            The index of the selected node.
        """

        # Compute the current embeddings
        current_embeddings = self.embed.cur_embed
        agg_embed = current_embeddings.sum(dim=0)  # Sum of all node embeddings

        if random.random() < EPSILON:
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

            agg_embed = state.sum(dim=0)  # Aggregate embedding for state
            node_embed = state[action]  # Embedding of the selected action node

            # Compute Q(s, a) for the current state-action pair
            q_value = self.q_network(node_embed.unsqueeze(0), agg_embed.unsqueeze(0))

            # Compute embeddings for the next state

            next_agg_embed = next_state.sum(dim=0)

            # Compute the target Q-value using max_a' Q(s', a')
            max_next_q = max(
                self.q_network(next_state.cur_embed[v].unsqueeze(0), next_agg_embed.unsqueeze(0))
                for v in range(self.graph.num_nodes) if next_state.graph.labels[v] != 1
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
        state_copy = state.clone().detach()
        next_state_copy = next_state.clone().detach()
        self.replay_buffer.append((state_copy, action, reward, next_state_copy))


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

        while not done:
            # Get the valid nodes (not yet selected)
            valid_nodes = [i for i, label in enumerate(env.embed.graph.labels) if label == 0]
            for i in range(env.embed.graph.num_nodes):
                if (env.embed.graph.labels[i]!=1):
                    valid_nodes.append(i)

            # Select an action 
            action = agent.select_action(valid_nodes)

            # Apply the action 
            next_state, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer
            agent.add_experience(state.cur_embed, action, reward, next_state.cur_embed)

            # Train the agent
            agent.train(batch_size)

            # Update the state
            state = next_state
            episode_reward += reward

        # Log episode performance
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {episode_reward}")


