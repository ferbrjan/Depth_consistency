import numpy as np
from matplotlib import pyplot as plt



X = np.load('test.npy')

std = X[:, :, 0]
mean = X[:, :, 1]

print(std)
print(mean)

fig, ax = plt.subplots()
im = plt.imshow(mean, interpolation='nearest')
fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()