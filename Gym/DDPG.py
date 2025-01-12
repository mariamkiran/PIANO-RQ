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

# Hyperparameters
REPLAY_CAPACITY = 2000
GAMMA = 0.99
LR_ALPHAS = 0.0005
LR_CRITIC = 0.001  # Learning rate for critic
LR_ACTOR = 0.0001  # Learning rate for actor  # MODIFIED
EPSILON = 0.1


class QNet(nn.Module):
    def __init__(self, embed_dim=32):
        super(QNet, self).__init__()
        self.embed_dim = embed_dim

        # Trainable parameters for critic (betas) and actor (thetas)  # MODIFIED
        self.beta1 = nn.Parameter(torch.rand(embed_dim*2, 1))  # Critic parameters
        self.beta2 = nn.Parameter(torch.rand(1))
        self.beta3 = nn.Parameter(torch.rand(1))

        self.theta1 = nn.Parameter(torch.rand(embed_dim*2, 1))  # Actor parameters
        self.theta2 = nn.Parameter(torch.rand(1))
        self.theta3 = nn.Parameter(torch.rand(1))

        # Final linear transformation
        #self.fc = nn.Linear(embed_dim * 2, 1)

    def forward(self, node_embed, agg_embed, role='critic'):  # MODIFIED
        if role == 'critic':
            scaled_aggregate = self.beta2 * agg_embed
            scaled_node = self.beta3 * node_embed
            final_weights = self.beta1
        elif role == 'actor':
            scaled_aggregate = self.theta2 * agg_embed
            scaled_node = self.theta3 * node_embed
            final_weights = self.theta1

        # Concatenate embeddings
        combined = torch.cat((scaled_aggregate, scaled_node), dim=1)  # Shape: [1, 2 * embed_dim]
        q_value = torch.matmul(F.relu(combined), final_weights) 
        return q_value


class DDPGAgent:  # MODIFIED
    def __init__(self, embed_dim=32):
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
        self.embed_dim = embed_dim

        self.alpha1 = nn.Parameter(torch.rand(1))
        self.alpha2 = nn.Parameter(torch.rand(1))
        self.alpha3 = nn.Parameter(torch.rand(1))
        self.alpha4 = nn.Parameter(torch.rand(1))
        self.shared_alphas = [self.alpha1, self.alpha2, self.alpha3, self.alpha4]

        # Actor and Critic networks  # MODIFIED
        self.actor = QNet(embed_dim)
        self.critic = QNet(embed_dim)

        # Optimizers for actor and critic  # MODIFIED
        self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(list(self.critic.parameters()), lr=LR_CRITIC)
        self.optimizer_alphas = optim.Adam(self.shared_alphas, lr=LR_ALPHAS)

    def select_action(self, env, valid_nodes, epsilon):
        current_embeddings = env.embed.cur_embed
        agg_embed = current_embeddings.sum(dim=0)  # Aggregate embedding

        if random.random() < epsilon:  # Exploration
            return random.choice(valid_nodes)
        else:  # Exploitation
            action_probs = []
            for v in valid_nodes:
                node_embed = current_embeddings[v]
                action_prob = self.actor(node_embed.unsqueeze(0), agg_embed.unsqueeze(0), role='actor')  # MODIFIED
                action_probs.append(action_prob)
            action_probs = torch.cat(action_probs).squeeze()
            return valid_nodes[action_probs.argmax().item()]
    
    def train(self, batch_size, gamma=0.99):  # MODIFIED
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        critic_loss_value = 0
        actor_loss_value = 0

        for state, action, reward, next_state in batch:
            agg_embed = state.cur_embed.sum(dim=0)
            node_embed = state.cur_embed[action]

            # Critic Loss (TD error)
            q_value = self.critic(node_embed.unsqueeze(0), agg_embed.unsqueeze(0), role='critic')
            next_agg_embed = next_state.cur_embed.sum(dim=0)

            next_q = []
            for v in range(state.graph.num_nodes):
                if next_state.graph.labels[v] == 1:  # Skip invalid nodes
                    next_q.append((0, v))
                else:
                    q_val = self.critic(next_state.cur_embed[v].unsqueeze(0), next_agg_embed.unsqueeze(0), role='critic').item()
                    next_q.append((q_val, v))

            max_next_q = max(next_q, key=lambda x: x[0])  # Get (max Q-value, best node)

            # Compute TD target
            target = reward + gamma * max_next_q[0]
            critic_loss_value += (q_value-target)**2

            # Actor Loss
            act_prob = []
            valid_mask = []

            for v in range(state.graph.num_nodes):
                if next_state.graph.labels[v] == 1:  # Skip invalid nodes
                    act_prob.append(torch.tensor([0.0], device=self.actor.beta1.device))  # Placeholder for invalid nodes
                    valid_mask.append(0)  # Mark as invalid
                else:
                    prob = self.actor(next_state.cur_embed[v].unsqueeze(0), next_agg_embed.unsqueeze(0), role='actor')
                    act_prob.append(prob.squeeze(1))  # Squeeze to ensure consistent shape
                    valid_mask.append(1)  # Mark as valid

            # Convert act_prob to a tensor
            act_prob = torch.cat(act_prob).to(self.actor.beta1.device)
            valid_mask = torch.tensor(valid_mask, dtype=torch.float32, device=self.actor.beta1.device)

            # Apply valid mask to act_prob
            act_prob = act_prob * valid_mask

            # Normalize probabilities for valid nodes
            act_prob = F.softmax(act_prob, dim=0)

            # Ground truth target for actor
            target = torch.zeros_like(act_prob, device=self.actor.beta1.device)
            target[max_next_q[1]] = 1  # Set the target for the best node

            # Compute actor loss
            actor_loss_value += F.cross_entropy(act_prob.unsqueeze(0), target.unsqueeze(0))



    def add_experience(self, state, action, reward, next_state):
        state_copy = state.copy_emb()
        next_state_copy = next_state.copy_emb()
        self.replay_buffer.append((state_copy, action, reward, next_state_copy))

    def evaluate(self, env, budget):
        start_time = time.time()
        env.embed.update()

        agg_embed = env.embed.cur_embed.sum(dim=0)
        q_list = []
        for v in range(env.embed.graph.num_nodes):
            node_embed = env.embed.cur_embed[v]
            q_value = self.actor(node_embed.unsqueeze(0), agg_embed.unsqueeze(0), role='actor').item()  # MODIFIED
            q_list.append((q_value, v))
        q_list.sort(reverse=True)

        for _, v in q_list[:budget]:  # Select top `budget` nodes
            env.embed.graph.labels[v] = 1

        result = simulate(env.embed.graph, 10000)

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
    

        


def train_agent(agent, env, episodes, batch_size, epsilon):
    """
    Trains the DDPG agent by interacting with the CustomEnv.

    Args:
        agent: Instance of DDPGAgent.
        env: Instance of CustomEnv.
        episodes: Number of episodes to train.
        batch_size: Batch size for training.
        epsilon: Exploration rate (passed explicitly for saving/loading continuity).
    """

    for episode in range(episodes):
        # Reset the environment to get the initial embeddings
        state = env.reset()  

        done = False
        episode_reward = 0

        while not done:
            # Get the valid nodes (not yet selected)
            valid_nodes = [i for i, label in enumerate(env.embed.graph.labels) if label == 0]

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
        print(f"Total influenced: {env.influence}")

    agent.replay_buffer.clear()

    # Save all parameters, including alphas, betas, thetas (via q_network), and epsilon
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),  # Save actor network
        'critic_state_dict': agent.critic.state_dict(),  # Save critic network
        'shared_alphas_state_dict': {f'alpha{i+1}': alpha for i, alpha in enumerate(agent.shared_alphas)},
    }, 'C:\\Users\\17789\\Desktop\\Graph Dataset\\DDPG_agent.pth')

    
def DDPG_main(num_nodes):
    input_file = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\subgraph1.txt'
    adj_list = {}

    # Initialize adjacency list
    for i in range(num_nodes + 1):
        adj_list[i] = []
    max_node = 0

    with open(input_file, 'r') as file:
        for line in file:
            u, v, weight = line.strip().split()  # Split into u, v, and weight
            u, v = int(u), int(v)  # Convert node IDs to integers
            weight = float(weight)  # Convert weight to float
            
            adj_list[u].append((v, weight))  # Add edge with weight
            max_node = max(max_node, max(u, v))

    if max_node < 100:
        return

    # Create a Graph object with the adjacency list
    graph = Graph(max_node + 1, adj_list)
    agent = DDPGAgent()
    epsilon = 0.10  # Default exploration rate

    # Load pre-trained model if it exists
    if os.path.exists('C:\\Users\\17789\\Desktop\\Graph Dataset\\DDPG_agent.pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('C:\\Users\\17789\\Desktop\\Graph Dataset\\DDPG_agent.pth')
        
        # Load Q-network (betas and thetas included)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Restore shared alphas
        shared_alphas_state_dict = checkpoint['shared_alphas_state_dict']
        for i, alpha in enumerate(agent.shared_alphas):
            alpha.data = shared_alphas_state_dict[f'alpha{i+1}']
        
        # Restore epsilon
        #epsilon = checkpoint.get('epsilon', 0.10)  # Use default if not saved
    else:
        print("No pre-trained agent found. Creating a new agent...")

    # Create the environment
    env = CustomEnv(graph, agent.shared_alphas, 10)

    # Train the agent
    train_agent(agent, env, 30, 10, epsilon)

    # Evaluate random selection strategy for comparison

    '''
    random_avg = 0.0
    for i in range(20):
        random_avg += agent.random_select(env, 10, max_node)
    print(f'Random result: {random_avg / 20}')

    # Evaluate the trained agent
    print(f'DDPG result: {agent.evaluate(env, 10)}')

    # Evaluate CELF for comparison
    print(f'CELF result: {celf(graph, 10)}')
    '''