import random
from DDPG import DDPG_main
from DQN import DQN_main
from gensubgraph2 import bfs_sample
import time

t = 0
while t < 20: 

      
      start_node = random.randint(1,6331)
      max_node = random.randint(300,400)

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


      if(runtime1>10):
            t+=1
      print(f"Runtime1: {runtime1:.2f} seconds")
      print(f"Runtime2: {runtime2:.2f} seconds")
    


