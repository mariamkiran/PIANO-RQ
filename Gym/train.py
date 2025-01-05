import random
from DQN import DQN_main
from gensubgraph import bfs_sample

for t in range (5):
    start_node = random.randint(5000,8000)
    max_node = random.randint(600,800)

    input_txt = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\wiki-Vote.txt'
    output_txt = 'C:\\Users\\17789\\Desktop\\Graph Dataset\\subgraph1.txt'
    bfs_sample(input_txt, output_txt, start_node, max_node)

    DQN_main(max_node)