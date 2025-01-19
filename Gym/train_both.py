import random
from DDPG import DDPG_main
from DQN import DQN_main
from gensubgraph2 import bfs_sample
import time
from AgentTest import test_main

t = 0

while t < 30: 
    
    if t%5 == 0 and t!=0:
         test_main(8274)    

    start_node = random.randint(1,8200)
    max_node = random.randint(500,800)

    input_txt = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\weighted_sample.txt'
    output_txt = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\subgraph1.txt'
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
    


