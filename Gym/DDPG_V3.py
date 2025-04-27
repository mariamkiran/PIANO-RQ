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
from simulator import simulate, celf
import torch.nn.utils as nn_utils

# Hyperparameters
REPLAY_CAPACITY = 64
GAMMA = 0.99
LR_ALPHAS = 0.0005
LR_CRITIC = 0.0010   # Learning rate for critic
LR_ACTOR = 0.0010    # Learning rate for actor
EPSILON = 0.20
GUMBEL_TAU = 0.75    # Temperature for Gumbel Softmax
TAU = 0.005          # Soft update interpolation factor

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
        self.fc2 = nn.Linear(embed_dim * 2, 1)  # Actor logits

        # Xavier init + weight-norm
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1 = nn_utils.weight_norm(self.fc1)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2 = nn_utils.weight_norm(self.fc2)

    def forward(self, node_embed, agg_embed, role='critic'):
        if role == 'critic':
            scaled_aggregate = self.beta2 * agg_embed
            scaled_node      = self.beta3 * node_embed
            combined         = torch.cat((scaled_aggregate, scaled_node), dim=1)
            return self.fc1(F.relu(combined))
        else:
            scaled_aggregate = self.theta2 * agg_embed
            scaled_node      = self.theta3 * node_embed
            combined         = torch.cat((scaled_aggregate, scaled_node), dim=1)
            return self.fc2(F.relu(combined))

class DDPGAgent:
    def __init__(self, embed_dim=64):
        self.replay_buffer = deque(maxlen=REPLAY_CAPACITY)
        self.embed_dim     = embed_dim

        # shared alphas for graph embedding update
        self.alpha1 = nn.Parameter(torch.rand(1))
        self.alpha2 = nn.Parameter(torch.rand(1))
        self.alpha3 = nn.Parameter(torch.rand(1))
        self.alpha4 = nn.Parameter(torch.rand(1))
        self.shared_alphas = [self.alpha1, self.alpha2, self.alpha3, self.alpha4]

        # actor & critic networks
        self.actor  = QNet(embed_dim)
        self.critic = QNet(embed_dim)
        # target networks (initialized to match the originals)
        self.actor_target  = QNet(embed_dim)
        self.critic_target = QNet(embed_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        # no gradient for targets
        for p in self.actor_target.parameters():  p.requires_grad = False
        for p in self.critic_target.parameters(): p.requires_grad = False

        # optimizers
        self.opt_actor  = optim.Adam(self.actor.parameters(),  
                                     lr=LR_ACTOR)
        self.opt_critic = optim.Adam(
            list(self.critic.parameters()) + self.shared_alphas,
            lr=LR_CRITIC
        )

    def select_action(self, env, valid_nodes, epsilon):
        cur = env.embed.cur_embed
        agg = cur.sum(dim=0, keepdim=True)
        if random.random() < epsilon:
            return random.choice(valid_nodes)
        v_emb  = cur[valid_nodes]
        a_emb  = agg.expand(len(valid_nodes), -1)
        logits = self.actor(v_emb, a_emb, role='actor').squeeze(1)
        logits = logits - logits.max()
        one_hot = F.gumbel_softmax(logits, tau=GUMBEL_TAU, hard=False, dim=-1)
        idx     = one_hot.argmax().item()
        return valid_nodes[idx]

    def add_experience(self, state, action, reward, next_state):
        self.replay_buffer.append((
            state.copy_emb(),
            action,
            reward,
            next_state.copy_emb()
        ))

    def train(self, batch_size, gamma=GAMMA):
        if len(self.replay_buffer) < batch_size:
            return

        # --- Critic update using target network ---
        self.opt_critic.zero_grad()
        critic_losses = []
        batch = random.sample(self.replay_buffer, batch_size)
        for s, a, r, ns in batch:
            agg_s = s.cur_embed.sum(0, keepdim=True)
            q_s   = self.critic(s.cur_embed[a].unsqueeze(0), agg_s, role='critic')

            agg_ns = ns.cur_embed.sum(0, keepdim=True)
            valid  = [i for i, lbl in enumerate(ns.graph.labels) if lbl == 0]
            # use critic_target for next‐state Q
            qn = self.critic_target(
                ns.cur_embed[valid],
                agg_ns.expand(len(valid), -1),
                role='critic'
            ).detach().max()
            target = r + gamma * qn
            critic_losses.append(F.smooth_l1_loss(q_s.squeeze(), target))

        loss_c = torch.stack(critic_losses).mean()
        print(f"[DEBUG] Critic loss = {loss_c.item():.4f}")
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic.parameters()) + self.shared_alphas,
            max_norm=10
        )
        self.opt_critic.step()

        # --- Actor update (unchanged TD target, but you could use target networks similarly) ---
        self.opt_actor.zero_grad()
        actor_losses = []
        for s, a, r, ns in batch:
            agg_s  = s.cur_embed.sum(0, keepdim=True)
            valid  = [i for i, lbl in enumerate(s.graph.labels) if lbl == 0]
            v_emb  = s.cur_embed[valid]
            a_emb  = agg_s.expand(len(valid), -1)

            logits = self.actor(v_emb, a_emb, role='actor').squeeze(1)
            logits = logits - logits.max()
            probs  = F.gumbel_softmax(logits, tau=GUMBEL_TAU, hard=False, dim=-1)
            logp   = torch.log(probs + 1e-8)

            agg_ns    = ns.cur_embed.sum(0, keepdim=True)
            valid_nxt = [i for i, lbl in enumerate(ns.graph.labels) if lbl == 0]
            # still using critic_target for stability
            qn = self.critic_target(
                ns.cur_embed[valid_nxt],
                agg_ns.expand(len(valid_nxt), -1),
                role='critic'
            ).detach().max()
            targ = r + gamma * qn

            advs = []
            for node in valid:
                qv = self.critic(
                    s.cur_embed[node].unsqueeze(0),
                    agg_s,
                    role='critic'
                ).detach()
                advs.append((targ - qv).squeeze())
            adv = torch.stack(advs)

            actor_losses.append(-(logp * adv).sum())

        loss_a = torch.stack(actor_losses).mean()
        print(f"[DEBUG] Actor loss = {loss_a.item():.4f}")
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=10
        )
        self.opt_actor.step()

        # --- Soft update target networks ---
        for src, tgt in zip(self.critic.parameters(), self.critic_target.parameters()):
            tgt.data.mul_(1.0 - TAU)
            tgt.data.add_(src.data * TAU)
        for src, tgt in zip(self.actor.parameters(), self.actor_target.parameters()):
            tgt.data.mul_(1.0 - TAU)
            tgt.data.add_(src.data * TAU)

    def evaluate(self, env, budget):
        start_time = time.time()
        env.embed.update()

        agg_embed = env.embed.cur_embed.sum(dim=0)
        q_list = []
        for v in range(env.embed.graph.num_nodes):
            node_embed = env.embed.cur_embed[v]
            q_value = self.critic(node_embed.unsqueeze(0), agg_embed.unsqueeze(0), role='actor').item()
            q_list.append((q_value, v))
        q_list.sort(reverse=True)

        for (q, v) in q_list:
            print(q)
            if budget <= 0:
                break
            env.embed.graph.labels[v] = 1
            budget -= 1
        
        result = simulate(env.embed.graph, 10000)

        end_time = time.time()
        print(f"DDPG: {end_time - start_time:.2f} seconds")

        env.reset()
        return result

    def random_select(self, env, nodes, num_nodes):
        env.embed.update()
        random_numbers = random.sample(range(num_nodes), nodes)
        for i in random_numbers:
            env.embed.graph.labels[i] = 1
        result = simulate(env.embed.graph, 1000)
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
            #print(f"[TRAIN] picked {action}, reward = {reward}")
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
    }, 'C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_3c2).pth')


    
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
    if os.path.exists('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_3c2).pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_3c2).pth')
        
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
    train_agent(agent, env, 10, 32, epsilon)

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