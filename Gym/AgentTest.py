from custom_graph import Graph
from DQN import DQNAgent
from DDPG import DDPGAgent
from simulator import simulate
from simulator import celf
from gymenv import CustomEnv
import os
import torch


def test_main(num_nodes):
    node_count = num_nodes
    input_file = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\weighted_sample.txt'
    adj_list = {}

    for i in range(node_count+1): adj_list[i]= []

    with open(input_file, 'r') as file:
        for line in file:
            u, v, weight = line.strip().split()  # Split into u, v, and weight
            u, v = int(u), int(v)  # Convert node IDs to integers
            weight = float(weight)  # Convert weight to float
            
            if u not in adj_list:
                adj_list[u] = []
            adj_list[u].append((v, weight))  # Add edge with weight

    # Create a Graph object with the adjacency list
    graph = Graph(node_count+1, adj_list)

    DQN_agent = DQNAgent()

    if os.path.exists('C:\\Users\\17789\\Desktop\\Graph Dataset\\DQN_agent.pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('C:\\Users\\17789\\Desktop\\Graph Dataset\\DQN_agent.pth')
        DQN_agent.q_network.load_state_dict(checkpoint['q_network_state_dict'])
    
        # Restore shared alphas
        shared_alphas_state_dict = checkpoint['shared_alphas_state_dict']
        for i, alpha in enumerate(DQN_agent.shared_alphas):
            alpha.data = shared_alphas_state_dict[f'alpha{i+1}']
    else:
        print("No pre-trained agent found. Creating a new agent...")

    DDPG_agent = DDPGAgent()

    if os.path.exists('C:\\Users\\17789\\Desktop\\Graph Dataset\\DDPG_agent.pth'):
        print("Loading pre-trained agent...")
        checkpoint = torch.load('C:\\Users\\17789\\Desktop\\Graph Dataset\\DDPG_agent.pth')
        
        # Load Q-network (betas and thetas included)
        DDPG_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        DDPG_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Restore shared alphas
        shared_alphas_state_dict = checkpoint['shared_alphas_state_dict']
        for i, alpha in enumerate(DDPG_agent.shared_alphas):
            alpha.data = shared_alphas_state_dict[f'alpha{i+1}']
        
        # Restore epsilon
        #epsilon = checkpoint.get('epsilon', 0.10)  # Use default if not saved
    else:
        print("No pre-trained agent found. Creating a new agent...")

    env = CustomEnv(graph, DQN_agent.shared_alphas, 10)

    #print(1)
    print(f'DQN: {DQN_agent.evaluate(env, 10)}')

    env = CustomEnv(graph, DDPG_agent.shared_alphas, 10)

    print(f'DDPG: {DDPG_agent.evaluate(env, 10)}')
    
    random_avg = 0.0
    for i in range(30):
        random_avg += DQN_agent.random_select(env,10,8274)
    print(f'Random: {random_avg/30}')
    print(f'CELF: {celf(graph,10)}')




    
test_main(8274)
