import random
import time
from DQN import DQN_main
from gensubgraph import bfs_sample

t = 0

while t < 5: 
    start_time = time.time()

    start_node = random.randint(5000,8000)
    max_node = random.randint(600,800)

    input_txt = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\wiki-Vote.txt'
    output_txt = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\subgraph1.txt'
    bfs_sample(input_txt, output_txt, start_node, max_node)

    DQN_main(max_node)

    end_time = time.time()
    runtime = end_time - start_time

    if(runtime>10):
         t+=1
    print(f"Runtime: {runtime:.2f} seconds")