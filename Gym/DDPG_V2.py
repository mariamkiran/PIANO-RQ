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

# Hyperparameters
REPLAY_CAPACITY = 2000
GAMMA = 0.99
LR_ALPHAS = 0.0005
LR_CRITIC = 0.0010  # Learning rate for critic
LR_ACTOR = 0.0010  # Learning rate for actor  # MODIFIED
EPSILON = 0.20
#torch.set_default_device('cuda')

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
        self.fc1 = nn.Linear(embed_dim * 2, 1)
        self.fc2 = nn.Linear(embed_dim * 2, 1)

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
        # q_value = torch.matmul(F.relu(combined), final_weights) 
        if role == 'critic':
            q_value = self.fc1(F.relu(combined))
            return q_value
        if role == 'actor':
            raw_prob = self.fc2(F.relu(combined))
            return raw_prob


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
        self.optimizer_critic = optim.Adam(
            list(self.critic.parameters()) + self.shared_alphas, 
            lr=LR_CRITIC
        )
        

    def select_action(self, env, valid_nodes, epsilon):
        current_embeddings = env.embed.cur_embed  # [num_nodes, embed_dim]
        agg_embed = current_embeddings.sum(dim=0)   # [embed_dim]

        if random.random() < epsilon:
            return random.choice(valid_nodes)
        else:
            valid_node_embeddings = current_embeddings[valid_nodes]  # [num_valid, embed_dim]
            # Here, since this is only used for selection (inference), expand is acceptable.
            agg_embed_batch = agg_embed.unsqueeze(0).repeat(len(valid_nodes), 1)  # [num_valid, embed_dim]
            action_values = self.actor(valid_node_embeddings, agg_embed_batch, role='actor')  # [num_valid, 1]
            action_values = action_values.squeeze(1)
            selected_index = torch.argmax(action_values).item()
            return valid_nodes[selected_index]
    
    def train(self, batch_size, gamma=0.99):
        if len(self.replay_buffer) < batch_size:
            return

        device = next(self.critic.parameters()).device
        critic_loss_total = torch.tensor(0.0, device=device, dtype=torch.float32)
        actor_loss_total = torch.tensor(0.0, device=device, dtype=torch.float32)

        batch = random.sample(self.replay_buffer, batch_size)
        
        for state, action, reward, next_state in batch:
            # ----- Critic Update -----
            # Current state: shape [1, embed_dim]
            agg_embed = state.cur_embed.sum(dim=0, keepdim=True)       # [1, embed_dim]
            node_embed = state.cur_embed[action].unsqueeze(0)            # [1, embed_dim]
            q_val = self.critic(node_embed, agg_embed, role='critic')      # [1, 1]
            
            # Next state: compute aggregate embedding
            next_agg_embed = next_state.cur_embed.sum(dim=0, keepdim=True) # [1, embed_dim]
            valid_next_nodes = [v for v in range(state.graph.num_nodes) if next_state.graph.labels[v] == 0]
            if valid_next_nodes:
                next_valid_embeds = next_state.cur_embed[valid_next_nodes]  # [num_valid, embed_dim]
                # Use expand instead of repeat:
                next_agg_batch = next_agg_embed.expand(len(valid_next_nodes), -1)  # [num_valid, embed_dim]
                q_next_vals = self.critic(next_valid_embeds, next_agg_batch, role='critic')  # [num_valid, 1]
                # Detach to prevent gradients flowing into the target:
                max_next_q = q_next_vals.detach().max()
            else:
                max_next_q = torch.tensor(0.0, device=q_val.device, dtype=q_val.dtype)
            
            # Compute TD target and detach it.
            target_value = reward + gamma * max_next_q
            # Ensure target has shape [1, 1] to match q_val:
            target = torch.tensor([[target_value]], device=q_val.device, dtype=q_val.dtype).detach()
            critic_loss = F.mse_loss(q_val, target)
            critic_loss_total += critic_loss

            # ----- Actor Update -----
            valid_current_nodes = [v for v in range(state.graph.num_nodes) if state.graph.labels[v] == 0]
            if valid_current_nodes:
                valid_embeds = state.cur_embed[valid_current_nodes]  # [num_valid, embed_dim]
                # Change repeat to expand:
                actor_agg_embed = agg_embed.expand(len(valid_current_nodes), -1)  # [num_valid, embed_dim]
                logits = self.actor(valid_embeds, actor_agg_embed, role='actor').squeeze()  # [num_valid]
                probs = F.softmax(logits, dim=0)
                log_probs = torch.log(probs + 1e-8)
                
                # Find index corresponding to the taken action.
                action_index = valid_current_nodes.index(action)
                advantage = (target - q_val).detach().squeeze()
                actor_loss = -log_probs[action_index] * advantage

                actor_loss_total += actor_loss

        # ----- Backpropagation -----
        self.optimizer_critic.zero_grad()
        critic_loss_total.backward()
        self.optimizer_critic.step()

        self.optimizer_actor.zero_grad()
        actor_loss_total.backward()
        self.optimizer_actor.step()




    def add_experience(self, state, action, reward, next_state):
        state_copy = state.copy_emb()
        next_state_copy = next_state.copy_emb()
        self.replay_buffer.append((state_copy, action, reward, next_state_copy))

    '''
    def evaluate(self, env, budget):
        #CELF Like algorithm
        start_time = time.time()
        env.embed.update()

        # Priority queue to store (-q_value, node, last_calculated_step)
        pq = []
        selected_count = 0
        current_step = 0  # Keeps track of when each node's probability was last calculated

        # Initial marginal gains (q_values)
        agg_embed = env.embed.cur_embed.sum(dim=0)
        for v in range(env.embed.graph.num_nodes):
            node_embed = env.embed.cur_embed[v]
            q_value = self.actor(node_embed.unsqueeze(0), agg_embed.unsqueeze(0), role='actor').item()
            heapq.heappush(pq, (-q_value, v, current_step))  # Push as negative for max-heap behavior

        # CELF-style selection
        while selected_count < budget and pq:
            neg_q_value, node, last_calculated_step = heapq.heappop(pq)

            # If node's value hasn't been recalculated in the current step, recompute it
            if last_calculated_step < current_step:
                node_embed = env.embed.cur_embed[node]
                updated_q_value = self.actor(node_embed.unsqueeze(0), agg_embed.unsqueeze(0), role='actor').item()
                heapq.heappush(pq, (-updated_q_value, node, current_step))  # Reinsert with updated value
            else:
                # Select the node
                env.embed.graph.labels[node] = 1
                selected_count += 1
                # Increment the current step for the next iteration
                current_step += 1
                env.embed.update()
                agg_embed = env.embed.cur_embed.sum(dim=0)

        # Simulate the result
        result = simulate(env.embed.graph, 10000)

        end_time = time.time()
        print(f"DDPG: {end_time - start_time:.2f} seconds")

        # Reset environment for next use
        env.reset()

        return result
    '''
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
            env.embed.graph.labels[v] = 1
            budget-=1
        
        result = simulate(env.embed.graph,10000)

        end_time = time.time()

        print(f"DDPG: {end_time - start_time:.2f} seconds")

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
        env.reset()  

        done = False
        episode_reward = 0
        
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
    }, 'C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_c2).pth')

    
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
    if os.path.exists('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_c2).pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_c2).pth')
        
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
    train_agent(agent, env, 15, 16, epsilon)

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