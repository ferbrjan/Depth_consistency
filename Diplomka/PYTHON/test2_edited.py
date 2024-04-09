import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import math
import mat73
from scipy.interpolate import griddata
import os


NUMBER_OF_IMAGES = 3
ORIG_WIDTH = 1920
ORIG_HEIGHT = 1440
DOWNSAMPLE_COEF = 3
NEW_WIDTH = int(ORIG_WIDTH/DOWNSAMPLE_COEF)
NEW_HEIGHT = int(ORIG_HEIGHT/DOWNSAMPLE_COEF)

def load_data(dir,num_images=3):
    img_files = os.listdir(dir + "/Image_data")
    ext_files = os.listdir(dir + "/Camera_extrinsics")
    img_files.sort()
    ext_files.sort()
    print("Loading files:")
    print(img_files)
    print(ext_files)

    RGBDDs = []
    extrinsics = []

    for img_file in img_files:
        if img_file.endswith(".mat"):
            data = mat73.loadmat(dir + "/Image_data/" + img_file)
            RGBDD = data['output_full']
            RGBDD = np.array(RGBDD)
            RGBDD = np.nan_to_num(RGBDD,
                                  nan=0)  # Replace NaNs with zeros??? or infinity???? Needs testing I guess but infinity seems more reasonable
            RGBDD = np.squeeze(RGBDD)
            RGBDD = torch.from_numpy(RGBDD).float()
            RGBDD = RGBDD.permute(0, 3, 1, 2)
            RGBDDs.append(RGBDD)

    for ext_file in ext_files:
        if ext_file.endswith(".mat"):
            data = mat73.loadmat(dir + "/Camera_extrinsics/" + ext_file)
            cam_params = data['output_params']
            cam_params = np.array(cam_params)
            cam_params = np.squeeze(cam_params)
            extrinsics.append(cam_params)
    print("Loading complete")
    return RGBDDs, extrinsics

class ModifiedCNN(nn.Module):
    def __init__(self):
        super(ModifiedCNN, self).__init__()

        # Initial convolution layers
        self.conv1 = nn.Conv2d(NUMBER_OF_IMAGES * 5, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Additional convolution layers with varying kernel sizes
        self.conv3_1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)  # 1x1 kernel
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, padding=2)  # Larger kernel

        # Final layer to map to desired output channels
        self.conv6 = nn.Conv2d(256, 1, kernel_size=3, padding=1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Convolutional layers with activation function and batch normalization
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Additional layers
        x = self.relu(self.bn3(self.conv3_1(x)))  # 1x1 convolution
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn4(self.conv5(x)))  # Using larger kernel

        x = self.conv6(x)  # Final convolution to get to the desired output size

        return x

RGBDDs, extrinsics = load_data('Exp2_data/3_images_solo_2', NUMBER_OF_IMAGES)

for i in range(len(RGBDDs)):
    orig = RGBDDs[i]
    x = torch.linspace(-1, 1, NEW_HEIGHT).view(-1, 1).repeat(1, NEW_WIDTH)
    y = torch.linspace(-1, 1, NEW_WIDTH).repeat(NEW_HEIGHT, 1)
    grid0 = torch.cat((y.unsqueeze(2), x.unsqueeze(2)), 2)
    grid0.unsqueeze_(0)

    grid = torch.from_numpy(np.tile(grid0, (NUMBER_OF_IMAGES, 1, 1, 1)))

    image_small = F.grid_sample(orig, grid)
    #plt.imshow(RGBDDs[i][0, 2, :, :].detach().numpy())
    #plt.show()
    #plt.imshow(image_small[0, 2, :, :].detach().numpy())
    #plt.show()

    RGBDDs[i] = image_small

net = ModifiedCNN()
PATH = 'model_single_frame_120_downsample_edited_withself_more_repros_2500iters_1090.pth'
net.load_state_dict(torch.load(PATH))


test_outputs = net(RGBDDs[0])


output = []
for i in range(NUMBER_OF_IMAGES):
    out = np.zeros((int(1440/DOWNSAMPLE_COEF), int(1920/DOWNSAMPLE_COEF)))
    out = test_outputs[i,0,:,:].detach().numpy()
    output.append(out)
    with open('result_GPU_12_1090_'+str(i)+'.npy', 'wb') as f:
        np.save(f, out)

#with open('result_GPU_040_1.npy', 'wb') as f:
    #np.save(f, out)


print("done")