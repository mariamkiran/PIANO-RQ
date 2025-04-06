import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
from collections import deque
import heapq
from custom_graph import Graph
from stovec import Embed
from gymenv import CustomEnv
from simulator import simulate
from simulator import celf
import torch.nn.utils as nn_utils

# Hyperparameters
REPLAY_CAPACITY = 64
GAMMA = 0.99
LR_ALPHAS = 0.0005
LR_CRITIC = 0.0010  # Learning rate for critic
LR_ACTOR = 0.0010   # Learning rate for actor
EPSILON = 0.20
GUMBEL_TAU = 0.75    # Temperature for Gumbel Softmax

class QNet(nn.Module):
    def __init__(self, embed_dim=64):
        super(QNet, self).__init__()
        self.embed_dim = embed_dim

        # Parameters for critic and actor
        self.beta1 = nn.Parameter(torch.rand(embed_dim * 2, 1))
        self.beta2 = nn.Parameter(torch.rand(1))
        self.beta3 = nn.Parameter(torch.rand(1))

        self.theta1 = nn.Parameter(torch.rand(embed_dim * 2, 1))
        self.theta2 = nn.Parameter(torch.rand(1))
        self.theta3 = nn.Parameter(torch.rand(1))

        # Final linear transformations
        self.fc1 = nn.Linear(embed_dim * 2, 1)  # Critic output
        self.fc2 = nn.Linear(embed_dim * 2, 1)
        nn.init.xavier_uniform_(self.fc1.weight)  # Xavier initialization
        self.fc1 = nn_utils.weight_norm(self.fc1)  # Apply weight normalization
        nn.init.xavier_uniform_(self.fc2.weight)  # Xavier initialization
        self.fc2 = nn_utils.weight_norm(self.fc2)  # Apply weight normalization

    def forward(self, node_embed, agg_embed, role='critic'):
        if role == 'critic':
            scaled_aggregate = self.beta2 * agg_embed
            scaled_node = self.beta3 * node_embed
            combined = torch.cat((scaled_aggregate, scaled_node), dim=1)
            q_value = self.fc1(F.relu(combined))
            return q_value
        elif role == 'actor':
            scaled_aggregate = self.theta2 * agg_embed
            scaled_node = self.theta3 * node_embed
            combined = torch.cat((scaled_aggregate, scaled_node), dim=1)
            logits = self.fc2(F.relu(combined))
            # Return raw logits (no sigmoid) for use with Gumbel Softmax
            return logits

class DDPGAgent:
    def __init__(self, embed_dim=64):
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
        self.embed_dim = embed_dim

        self.alpha1 = nn.Parameter(torch.rand(1))
        self.alpha2 = nn.Parameter(torch.rand(1))
        self.alpha3 = nn.Parameter(torch.rand(1))
        self.alpha4 = nn.Parameter(torch.rand(1))
        self.shared_alphas = [self.alpha1, self.alpha2, self.alpha3, self.alpha4]

        self.actor = QNet(embed_dim)
        self.critic = QNet(embed_dim)

        self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=LR_ACTOR)
        self.optimizer_critic = optim.Adam(list(self.critic.parameters()) + self.shared_alphas, lr=LR_CRITIC)

    def select_action(self, env, valid_nodes, epsilon):
        current_embeddings = env.embed.cur_embed
        agg_embed = current_embeddings.sum(dim=0)

        if random.random() < epsilon:
            return random.choice(valid_nodes)
        else:
            # Get embeddings for valid nodes: shape (num_valid, embed_dim)
            valid_node_embeds = current_embeddings[valid_nodes]
            agg_embed_batch = agg_embed.unsqueeze(0).repeat(len(valid_nodes), 1)
            # Get raw logits from the actor network
            logits = self.actor(valid_node_embeds, agg_embed_batch, role='actor').squeeze(1)
            logits = logits - logits.max()
            # (No subtraction of logits.max() here)
            # Apply Gumbel Softmax (hard=False to keep it differentiable)
            if torch.isnan(logits).any():
                print("DEBUG: logits contain NaN")

            if torch.isinf(logits).any():
                print("DEBUG: logits contain Inf")

            gumbel_probs = F.gumbel_softmax(logits, tau=GUMBEL_TAU, hard=False, dim=0)

            if torch.isnan(gumbel_probs).any():
                print("DEBUG: logits contain NaN")

            if torch.isinf(gumbel_probs).any():
                print("DEBUG: logits contain Inf")
            # Use multinomial sampling for exploration
            sampled_index = torch.multinomial(gumbel_probs, num_samples=1).item()
            return valid_nodes[sampled_index]

    def add_experience(self, state, action, reward, next_state):
        state_copy = state.copy_emb()
        next_state_copy = next_state.copy_emb()
        self.replay_buffer.append((state_copy, action, reward, next_state_copy))

    def train(self, batch_size, gamma=GAMMA):
        if len(self.replay_buffer) < batch_size:
            return

        # --- Critic Training ---
        critic_losses = []
        batch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state in batch:
            # Current state embeddings
            agg_embed = state.cur_embed.sum(dim=0, keepdim=True)
            node_embed = state.cur_embed[action].unsqueeze(0)
            q_val = self.critic(node_embed, agg_embed, role='critic')
            
            # Next state: use all valid nodes (assumes at least one valid node exists)
            next_agg_embed = next_state.cur_embed.sum(dim=0, keepdim=True)
            valid_next_nodes = [v for v in range(state.graph.num_nodes) if next_state.graph.labels[v] == 0]
            next_valid_embeds = next_state.cur_embed[valid_next_nodes]
            next_agg_batch = next_agg_embed.expand(len(valid_next_nodes), -1)
            q_next_vals = self.critic(next_valid_embeds, next_agg_batch, role='critic')
            max_next_q = q_next_vals.detach().max()
            
            target_value = reward + gamma * max_next_q
            target = torch.tensor([[target_value]], device=q_val.device, dtype=q_val.dtype)
            critic_loss = F.mse_loss(q_val, target)
            critic_losses.append(critic_loss)
            
        critic_loss_total = torch.stack(critic_losses).mean()
        self.optimizer_critic.zero_grad()
        critic_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10)
        self.optimizer_critic.step()

        # --- Actor Training ---
        actor_losses = []
        for state, action, reward, next_state in batch:
            agg_embed = state.cur_embed.sum(dim=0, keepdim=True)
            valid_current_nodes = [v for v in range(state.graph.num_nodes) if state.graph.labels[v] == 0]
            valid_embeds = state.cur_embed[valid_current_nodes]
            actor_agg_embed = agg_embed.expand(len(valid_current_nodes), -1)
            # Compute raw logits from actor (no normalization)
            logits = self.actor(valid_embeds, actor_agg_embed, role='actor').squeeze(1)
            logits = logits - logits.max()
            # Use Gumbel Softmax to obtain probability distribution
            probs = F.gumbel_softmax(logits, tau=GUMBEL_TAU, hard=False, dim=0)
            log_probs = torch.log(probs + 1e-8)
            
            # Compute the TD target from the next state (same for all actions in this state)
            next_agg_embed = next_state.cur_embed.sum(dim=0, keepdim=True)
            valid_next_nodes = [v for v in range(state.graph.num_nodes) if next_state.graph.labels[v] == 0]
            next_valid_embeds = next_state.cur_embed[valid_next_nodes]
            next_agg_batch = next_agg_embed.expand(len(valid_next_nodes), -1)
            q_next_vals = self.critic(next_valid_embeds, next_agg_batch, role='critic')
            max_next_q = q_next_vals.detach().max()
            target_value = reward + gamma * max_next_q
            
            # Compute advantage for each valid action as: advantage = target_value - Q(s, node)
            adv_list = []
            for idx in range(len(valid_current_nodes)):
                q_val_node = self.critic(state.cur_embed[valid_current_nodes[idx]].unsqueeze(0),
                                           agg_embed, role='critic').detach()
                adv = target_value - q_val_node
                adv_list.append(adv)
            adv_vector = torch.stack(adv_list).squeeze()  # shape: (num_valid,)
            
            # Actor loss: sum over valid actions of -log(prob) * (advantage)
            sample_policy_loss = (- (log_probs * adv_vector)).sum()

            lambda_reg = 0.1  # Tune this hyperparameter
            range_penalty = lambda_reg * ((F.relu(0 - logits))**2 + (F.relu(logits - 100))**2).mean()

            sample_actor_loss = sample_policy_loss + range_penalty
            actor_losses.append(sample_actor_loss)

            
            
        actor_loss_total = torch.stack(actor_losses).mean()
        self.optimizer_actor.zero_grad()
        actor_loss_total.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
        self.optimizer_actor.step()

    def evaluate(self, env, budget):
        start_time = time.time()
        env.embed.update()

        agg_embed = env.embed.cur_embed.sum(dim=0) 
        #NODE NUMBER MUST START WITH ZERO!
        q_list = []
        for v in range(env.embed.graph.num_nodes):
            node_embed = env.embed.cur_embed[v]
            q_value = self.actor(node_embed.unsqueeze(0), agg_embed.unsqueeze(0),role='actor').item()
            q_list.append((q_value,v))
        q_list.sort(reverse=True)

        for (q,v) in q_list:
            if budget<=0:
                break
            print(q)
            env.embed.graph.labels[v] = 1
            budget-=1
        
        result = simulate(env.embed.graph,10000)

        end_time = time.time()

        print(f"DDPG: {end_time - start_time:.2f} seconds")

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
        env.reset()  

        done = False
        episode_reward = 0
        
        if episode == episodes - 1:
            epsilon = 0

        while not done:
            # Get the valid nodes (not yet selected)
            valid_nodes = [i for i, label in enumerate(env.embed.graph.labels) if label == 0]
            
            orig_state = env.embed.copy_emb()
            # Select an action 
            action = agent.select_action(env, valid_nodes, epsilon)
            
            
            # Apply the action 
            state, reward, done, _ = env.step(action)

            # Add the experience to the replay buffer
            agent.add_experience(orig_state, action, reward, state)

            # Train the agent
            agent.train(batch_size)

            episode_reward += reward
            

        epsilon *= 0.90
        # Log episode performance
        print(f"Episode {episode + 1}/{episodes} - Total Reward: {episode_reward}")
        print(f"Total influenced: {env.influence}")

    #agent.replay_buffer.clear()

    # Save all parameters, including alphas, betas, thetas (via q_network), and epsilon
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),  # Save actor network
        'critic_state_dict': agent.critic.state_dict(),  # Save critic network
        'shared_alphas_state_dict': {f'alpha{i+1}': alpha for i, alpha in enumerate(agent.shared_alphas)},
    }, 'C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_4c4).pth')


    
def DDPG_main(num_nodes):


    input_file = 'C:\\Users\\17789\\Desktop\\New Graph Dataset\\subgraph1.txt'
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
    epsilon = 0.30  # Default exploration rate

    # Load pre-trained model if it exists
    if os.path.exists('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_4c4).pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_4c4).pth')
        
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
    train_agent(agent, env, 10, 16, epsilon)

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