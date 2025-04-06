import torch
params = torch.load('C:\\Users\\17789\\Desktop\\New Graph Dataset\\DDPG_agent(p2p1_4c3).pth', map_location="cpu")
for key, value in params.items():
    print(f"{key}: {value}")