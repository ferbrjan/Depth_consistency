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
import os
import torch.nn as nn
import argparse

NUMBER_OF_IMAGES = 3
ORIG_WIDTH = 1920
ORIG_HEIGHT = 1440
DOWNSAMPLE_COEF=3
NEW_WIDTH = int(ORIG_WIDTH/DOWNSAMPLE_COEF)
NEW_HEIGHT = int(ORIG_HEIGHT/DOWNSAMPLE_COEF)

parser = argparse.ArgumentParser(description="Conv_apporach")

# Adding arguments
parser.add_argument("data_path", type=str, help="Input path for the data")
parser.add_argument("k_path", type=str, help="Input path for the file with matrix K")
parser.add_argument("PATH", type=str, help="Output path for the model")

# Parse the arguments
args = parser.parse_args()

def interpolate(grids, offsets):
    b, c, h, w = grids.shape
    #normalized_offsets = offsets / torch.tensor([(w - 1) / 2, (h - 1) / 2], device=grids.device)
    normalized_offsets = 2 * offsets / torch.tensor([w - 1, h - 1], device=grids.device) - 1
    # Reshape offsets for grid sampling
    repeated_offsets = normalized_offsets.view(b, h, w, 2).float()

    # Grid sampling
    interpolated_desc = F.grid_sample(grids.float(), repeated_offsets, mode='bilinear', align_corners=True)

    #print(interpolated_desc.shape)
    #print(interpolated_desc[0,0,0,:])
    return interpolated_desc
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



def a2h(x, w=1):
    if w is None:
        w = 1
    X = np.vstack([x, w * np.ones((1, x.shape[1]))])
    return X

def h2a(X, w=None):
    x = X[:-1, :]
    xe = X[-1, :]
    if w is None:
        x = x / (np.ones((x.shape[0], 1)) * xe)
    else:
        w = np.asarray(w).flatten()
        if any(w):
            x[:, w] = x[:, w] / (np.ones((x.shape[0], 1)) * xe[w])
    return x

def reproject(x, y, d, E_ref, E_src, K):
    #h2a(K * h2a(E_src * inv(E_ref) *a2h(inv(K) * a2h([x(:),y(:)]') .* repmat(d(:)',3,1))))

    x = x.flatten()
    y = y.flatten()
    d = d.detach().numpy()
    d = d.flatten()
    xy = np.vstack([y, x])
    xy = a2h(xy)
    Kxy = np.linalg.inv(K) @ xy
    rep = np.matlib.repmat(d, 3, 1)
    right_side = a2h(Kxy * rep)
    return(h2a(K@ (h2a(E_src @ np.linalg.inv(E_ref) @ right_side))))



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

# Read mat files
RGBDDs, extrinsics = load_data(args.data_path, NUMBER_OF_IMAGES)
#Camera intrinsics
data = mat73.loadmat(args.k_path)
K = data['K']
K = K/DOWNSAMPLE_COEF
K[2,2] = 1


#Downsample
for i in range(len(RGBDDs)):
    orig = RGBDDs[i]
    x = torch.linspace(-1, 1, NEW_HEIGHT).view(-1, 1).repeat(1, NEW_WIDTH)
    y = torch.linspace(-1, 1, NEW_WIDTH).repeat(NEW_HEIGHT, 1)
    grid0 = torch.cat((y.unsqueeze(2), x.unsqueeze(2)), 2)
    grid0.unsqueeze_(0)
    grid = torch.from_numpy(np.tile(grid0, (NUMBER_OF_IMAGES, 1, 1, 1)))

    image_small = F.grid_sample(orig, grid)
    RGBDDs[i] = image_small


# Define  network, loss function and optimizer
net = ModifiedCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("training on device",device)
net.to(device)
print(torch.cuda.memory_allocated())

# Train network
num_epochs = 2500
losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    print("Epoch %d \n\n" % epoch)
    #Add another for loop with more images here
    for batch in range(len(RGBDDs)):
        RGBDD = RGBDDs[batch]
        RGBDD = RGBDD.to(device)
        cam_params = extrinsics[batch]
        print("Training for image batch %d of total %d" % (batch+1, len(RGBDDs)))

        batch_loss = 0
        optimizer.zero_grad()
        outputs = net(RGBDD)
        for i, data in enumerate(RGBDD, 0): #Training for one batch
            x, y = np.meshgrid(np.arange(0.5, NEW_WIDTH+0.5), np.arange(0.5, NEW_HEIGHT+0.5))
            for j in range(NUMBER_OF_IMAGES):
                if i != j:
                    uv_from_ref = reproject(x, y, RGBDD[j][1].cpu(), cam_params[j], cam_params[i], K) #reproject i to j - THIS WORKS, I checked matlab version
                    uv_from_ref = uv_from_ref[::-1]
                    uv_from_ref_reshaped = uv_from_ref.reshape((2,NEW_HEIGHT, NEW_WIDTH)).transpose(1,2,0)
                    uv_from_ref_reshaped = torch.from_numpy(uv_from_ref_reshaped.copy())
                    uv_from_ref_reshaped = uv_from_ref_reshaped.to(device)

                    channel_to_interpolate = outputs[i][0].float()
                    channel_to_interpolate = channel_to_interpolate[None, None, :, :]
                    channel_to_interpolate = channel_to_interpolate.to(device)

                    ref_from_src_depth_ios = interpolate(channel_to_interpolate,uv_from_ref_reshaped)
                    ref_from_src_depth_ios = ref_from_src_depth_ios.to(device)
                else:
                    ref_from_src_depth_ios = (outputs[i].unsqueeze(0)).float()
                    ref_from_src_depth_ios = ref_from_src_depth_ios.to(device)

                """
                fig, ax = plt.subplots()
                im = plt.imshow(RGBDD[i, 3, :, :].detach().cpu().numpy())
                fig.colorbar(im, ax=ax, label='Interactive colorbar')
                fig.suptitle('Transformed image', fontsize=16)
                plt.show()

                fig, ax = plt.subplots()
                im = plt.imshow(RGBDD[j, 3, :, :].detach().cpu().numpy())
                fig.colorbar(im, ax=ax, label='Interactive colorbar')
                fig.suptitle('Transformed image', fontsize=16)
                plt.show()


                fig, ax = plt.subplots()
                im = plt.imshow(ref_from_src_depth_ios[0, 0, :, :].detach().cpu().numpy())
                fig.colorbar(im, ax=ax, label='Interactive colorbar')
                fig.suptitle('Transformed image', fontsize=16)
                plt.show()
                """
                orig = RGBDD[j,1,:,:]
                generated = ref_from_src_depth_ios[0,0,:,:]
                orig = orig.to(device)
                generated = generated.to(device)
                mask = (generated != 0).float() #find relevant (nonzero) values

                loss = criterion(orig*mask,generated*mask)
                batch_loss += loss



        #Reproject outputs into each other
        x, y = np.meshgrid(np.arange(0.5, NEW_WIDTH + 0.5), np.arange(0.5, NEW_HEIGHT + 0.5))
        output_reprojections_loss = 0
        for i in range(len(outputs)):
            for j in range(len(outputs)):
                if i != j:
                    uv_from_ref = reproject(x, y, outputs[j, 0, :, :].float().cpu(), cam_params[j], cam_params[i], K)
                    uv_from_ref = uv_from_ref[::-1]
                    uv_from_ref_reshaped = uv_from_ref.reshape((2, NEW_HEIGHT, NEW_WIDTH)).transpose(1, 2, 0)
                    uv_from_ref_reshaped = torch.from_numpy(uv_from_ref_reshaped.copy())
                    uv_from_ref_reshaped = uv_from_ref_reshaped.to(device)

                    channel_to_interpolate = outputs[i, 0, :, :].float()
                    channel_to_interpolate = channel_to_interpolate[None, None, :, :]

                    ref_from_src_depth_ios = interpolate(channel_to_interpolate, uv_from_ref_reshaped)
                    ref_from_src_depth_ios = ref_from_src_depth_ios.to(device)

                    orig = outputs[j, 0, :, :]
                    generated = ref_from_src_depth_ios[0, 0, :, :]
                    orig = orig.to(device)
                    generated = generated.to(device)
                    mask = (generated != 0).float()  # find relevant (nonzero) values

                    loss = criterion(orig * mask, generated * mask)
                    output_reprojections_loss += loss




        full_loss = 0.1*batch_loss + 0.9*output_reprojections_loss
        print("The batch loss is now",batch_loss)
        print("The output reprojections loss is now",output_reprojections_loss)
        print("The full loss is now",full_loss)
        print("")
        full_loss.backward()
        optimizer.step()
        running_loss += full_loss.item()
        #print("The running loss is now",running_loss)
        losses.append(full_loss.item())

        if (epoch%10==1):
            plt.plot(losses)
            plt.show()


print('Finished Training')

plt.plot(losses)
plt.savefig('graph.pdf')



# Save network
#PATH = 'model_single_frame_120_downsample_edited_withself_more_repros_2500iters_1090.pth' #0604 before
torch.save(net.state_dict(), args.PATH)



