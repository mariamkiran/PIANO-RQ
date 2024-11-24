import gym
from gym import spaces
import numpy as np
from custom_graph import Graph
from stovec import Embed
from simulator import simulate

class CustomEnv(gym.Env):

    def __init__(self, graph, budget):
        super(CustomEnv, self).__init__()

        #how many nodes can current seed set influence
        #initialize to 0 b/c seed set is empty
        self.influence = 0

        #how many seed nodes to select
        self.budget = budget
        self.num_step = 0

        #embeddings (2d vector representation of the graph)
        self.embed = Embed(graph)

        #action = change the state of a node
        self.action_space = spaces.Discrete(self.embed.graph.num_nodes)

        #observation is the embeddings
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=np.float32)


    def reset(self):
        self.influence = 0
        self.embed = Embed(self.embed.graph)
        return self.embed
    

    def step(self, action):
        if self.embed.graph.labels[action]==1:
            return self.embed.cur_embed, -10, True, {}

        #Default Iter set to 1000
        #Calculate reward
        new_inf =  simulate(self.embed.graph, 1000) 
        marginal_gain = max(0,new_inf-self.influence)

        #calculate if done
        self.num_step+=1
        done = (self.num_step>=self.budget)

        #update embeddings (ovservation)
        self.embed.graph.labels[action]==1
        self.embed.update()

        return self.embed, marginal_gain, done, {}
    
    
