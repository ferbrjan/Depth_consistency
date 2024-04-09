import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import matplotlib.pyplot as plt
import numpy as np
import math

# Read the .mat files
# Mean = 5+
mat150 = scipy.io.loadmat('00150_data.mat')
means150 = mat150["means"][0]
means150 = torch.from_numpy(means150).float()
stds150 = mat150["stds"][0]
stds150 = torch.from_numpy(stds150).float()
inputs150 = mat150["softmax_out"]
inputs150 = torch.from_numpy(inputs150).float()
# Mean = 3+
mat80 = scipy.io.loadmat('00080_data.mat')
means80 = mat80["means"][0]
means80 = torch.from_numpy(means80).float()
stds80 = mat80["stds"][0]
stds80 = torch.from_numpy(stds80).float()
inputs80 = mat80["softmax_out"]
inputs80 = torch.from_numpy(inputs80).float()
# Mean = 2+
mat20 = scipy.io.loadmat('00020_data.mat')
means20 = mat20["means"][0]
means20 = torch.from_numpy(means20).float()
stds20 = mat20["stds"][0]
stds20 = torch.from_numpy(stds20).float()
inputs20 = mat20["softmax_out"]
inputs20 = torch.from_numpy(inputs20).float()
# Mean = 5+
mat180 = scipy.io.loadmat('00180_data.mat')
means180 = mat180["means"][0]
means180 = torch.from_numpy(means180).float()
stds180 = mat180["stds"][0]
stds180 = torch.from_numpy(stds180).float()
inputs180 = mat180["softmax_out"]
inputs180 = torch.from_numpy(inputs180).float()

print(means150.shape, stds150.shape, inputs150.shape)

#Concatenate into training data and labels
means = torch.cat((means150,means80[:len(means150)],means20[:len(means150)],means180),0)
stds = torch.cat((stds150,stds80[:len(stds150)],stds20[:len(means150)],stds180),0)
inputs = torch.cat((inputs150,inputs80[:len(inputs150)],inputs20[:len(means150)],inputs180),0)

# Shuffle the data
idx = torch.randperm(len(means))
means = means[idx]
stds = stds[idx]
inputs = inputs[idx]

print(means.shape, stds.shape, inputs.shape)


# Define the network
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

def KL_divergence(std0, std1, mean0, mean1):
    diff_std = torch.abs(std0 - std1)
    diff_mean = torch.abs(mean0 - mean1)

    #std1 = abs(std1)
    #std0 = abs(std0)
    #pq = torch.log(std1 / (std0+0.00000001)) + (std0 ** 2 + (mean0 - mean1) ** 2) / (2 * std1 ** 2) #- 0.5
    #qp = torch.log(std0 / (std1+0.00000001)) + (std1 ** 2 + (mean1 - mean0) ** 2) / (2 * std0 ** 2) #- 0.5
    #print(torch.log(std1 / (std0+0.00000001)))
    #print((std0 ** 2 + (mean0 - mean1) ** 2) / (2 * std1 ** 2))
    #print(pq,qp)
    #print("\n\n")

    return diff_std + 0.05*diff_mean #+ 0.1*(pq) #pq+qp #diff #0.1*(pq + qp) + 0.9*(diff)



# Define the loss function
optimizer = optim.SGD(net.parameters(), lr=0.01)

losses = []
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(len(inputs)):
        # get the inputs; data is a list of [inputs, labels]
        inp = inputs[i,:]
        mean = means[i]
        std = stds[i]
        #print(mean,std)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = net(inp)
        #print(output)
        loss = KL_divergence(output[0], std, output[1], mean)
        loss.backward()
        optimizer.step()

        # print statistics
        #print(loss.item())
        #print("\n\n")
        running_loss += loss.item()
        losses.append(loss.item())
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

plt.plot(losses[2000:])
plt.show()

PATH = 'v4_net.pth'
torch.save(net.state_dict(), PATH)


