import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import matplotlib.pyplot as plt
import numpy as np
import mat73
import random

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


random_pairs = [(random.randint(0, 1399), random.randint(0, 1919)) for _ in range(10)]

data = scipy.io.loadmat("original_00020.mat")
print(data.keys())
depth = data["depth"]


for row in range(len(mat['data'])):
    print(row, "out of 1440")
    for col in range(len(mat['data'][row])):
        test =torch.from_numpy(mat['data'][row][col]).float()
        res = net(test)
        out[row][col][0] = res[0]
        out[row][col][1] = res[1]
        if (row, col) in random_pairs:
            print("Row: ", row, "Col: ", col, "Res: ", res, "Data", mat['data'][row][col], "Orig", depth[row][col])

            # Plotting
            fig, ax = plt.subplots()

            # Bar plot for softmax probabilities
            bars = ax.bar(range(len(mat['data'][row][col])), mat['data'][row][col], color='green', alpha=0.7)

            # Vertical line for original mean
            ax.axvline(x=depth[row][col]*10, color='b', label='LiDAR depth', linewidth=2)

            # Vertical line for estimated mean
            ax.axvline(x=res[1].detach().numpy()*10, color='r', label='Estimated Mean', linewidth=2)

            # Vertical lines for standard deviation
            ax.axvline(x=res[1].detach().numpy()*10 - res[0].detach().numpy()*10, color='r', linestyle='--', label='Std Deviation')
            ax.axvline(x=res[1].detach().numpy()*10 + res[0].detach().numpy()*10, color='r', linestyle='--')

            # Adding legend
            ax.legend()

            # Labels and title
            ax.set_xlabel('Distance [cm]')
            ax.set_xlim(res[1].detach().numpy()*10-25, res[1].detach().numpy()*10+25)
            ax.set_ylabel('Probability [%]')
            #ax.set_title('Softmax Vector Visualization')

            # Show plot
            plt.show()



with open('results/test.npy', 'wb') as f:
    np.save(f, out)


