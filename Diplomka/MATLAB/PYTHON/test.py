import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy


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

# Read the .mat file
mat2 = scipy.io.loadmat('00180_data.mat')

test_means = mat2["means"][0]
test_means = torch.from_numpy(test_means).float()
test_stds = mat2["stds"][0]
test_stds = torch.from_numpy(test_stds).float()
test_inputs = mat2["softmax_out"]
test_inputs = torch.from_numpy(test_inputs).float()

net = Net(256, 128, 2)
PATH = './v2_net.pth'
net.load_state_dict(torch.load(PATH))

test_outputs = net(test_inputs)

for i in range(len(test_outputs)):
    print(test_outputs[i], test_means[i], test_stds[i])
