import random
from DDPG import DDPG_main
from DQN import DQN_main
from gensubgraph import bfs_sample
import time

t = 0

while t < 10: 
    

    start_node = random.randint(5000,8000)
    max_node = random.randint(600,800)

    input_txt = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\wiki-Vote.txt'
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