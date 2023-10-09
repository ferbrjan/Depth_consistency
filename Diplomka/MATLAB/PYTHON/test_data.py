import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import matplotlib.pyplot as plt
import numpy as np
import mat73

class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 256 -> 128
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 128 -> 128
        self.fc3 = nn.Linear(hidden_size, output_size)  # 128 -> 2


    def forward(self, x):
        x = F.relu(self.fc1(x))  # 256 -> 128
        x = F.relu(self.fc2(x))  # 128 -> 128
        x = self.fc3(x)  # 128 -> 2
        return x

net = Net(256, 128, 2)
PATH = './v3_net.pth'
net.load_state_dict(torch.load(PATH))

mat = mat73.loadmat('00020_test.mat')

print(mat.keys())
print(len(mat['data']))
print(len(mat['data'][0]))

out = np.zeros((len(mat['data']), len(mat['data'][0]), 2))

for row in range(len(mat['data'])):
    print(row, "out of 1440")
    for col in range(len(mat['data'][row])):
        test =torch.from_numpy(mat['data'][row][col]).float()
        res = net(test)
        out[row][col][0] = res[0]
        out[row][col][1] = res[1]



with open('test.npy', 'wb') as f:
    np.save(f, out)


