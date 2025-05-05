import random
from Gym.DDPG_V3 import DDPG_main
from Gym.DQN_V2 import DQN_main
from Gym.gensubgraph2 import bfs_sample
import time
from tests.graph_test import build_graph_from_txt

t = 0

#graph details to help it select sub graph
input_graph_file="graph_datasets/DBLP-new.txt"


while t < 10 : 

      
      start_node = random.randint(1,6300)
      max_node = random.randint(300,450)
      graph=build_graph_from_txt(input_graph_file)
      num_nodes = graph.number_of_nodes()
      num_edges = graph.number_of_edges()
      print(f"Number of nodes: {num_nodes}")
      print(f"Number of edges: {num_edges}")

      nodes = graph.nodes
      print(nodes)
      # Randomly select a node
      random_node = random.choice(nodes)
      print("random:")
      # Print the selected node
      print(random_node)

      input_txt = "C:\\Users\\17789\\Desktop\\New Graph Dataset\\p2p(1).txt"
      output_txt = "C:\\Users\\17789\\Desktop\\New Graph Dataset\\subgraph1.txt"
      print(t)
      bfs_sample(input_txt, output_txt, start_node, max_node)

      start_time = time.time()
      DDPG_main(max_node)
      end_time1 = time.time()

      DQN_main(max_node)
      end_time2 = time.time()


      runtime1 = end_time1 - start_time
      runtime2 = end_time2 - end_time1


      if(runtime2>10):
            t+=1
      print(f"Runtime1: {runtime1:.2f} seconds")
      print(f"Runtime2: {runtime2:.2f} seconds")
    


