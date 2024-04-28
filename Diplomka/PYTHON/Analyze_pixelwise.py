import numpy as np
from matplotlib import pyplot as plt
import mat73
import scipy

data = scipy.io.loadmat("00020.mat")
print(data.keys())
depth = data["depth"]

X = np.load('results/test.npy')

std = X[:, :, 0]
mean = X[:, :, 1]
#print(std)
#print(mean)

fig, ax = plt.subplots()
im = plt.imshow(depth, interpolation='nearest')
fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()

# MSE
mse =  np.mean((depth - mean)**2)

# MAE
mae = np.mean(np.abs(depth - mean))

#Thresholding
diff01 = np.abs(depth - mean)
thresh1 = diff01
thresh1[diff01 > 0.1] = 0
thresh1[diff01 > 0] = 1


diff02 = np.abs(depth - mean)
thresh2 = diff02
thresh2[diff02 > 0.2] = 0
thresh2[diff02 > 0] = 1


fig, ax = plt.subplots()
im = plt.imshow(thresh1, interpolation='nearest')
#fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()

fig, ax = plt.subplots()
im = plt.imshow(thresh2, interpolation='nearest')
#fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()

print("MSE: ", mse)
print("MAE: ", mae)
