import torch
import numpy as np
import numpy.matlib
import mat73
import os
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import cv2


NUMBER_OF_IMAGES = 3
ORIG_WIDTH = 1920
ORIG_HEIGHT = 1440
DOWNSAMPLE_COEF=3
NEW_WIDTH = int(ORIG_WIDTH/DOWNSAMPLE_COEF)
NEW_HEIGHT = int(ORIG_HEIGHT/DOWNSAMPLE_COEF)

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

RGBDDs, extrinsics = load_data('Exp2_data/test_012', NUMBER_OF_IMAGES)

for i in range(len(RGBDDs)):
    orig = RGBDDs[i]
    x = torch.linspace(-1, 1, NEW_HEIGHT).view(-1, 1).repeat(1, NEW_WIDTH)
    y = torch.linspace(-1, 1, NEW_WIDTH).repeat(NEW_HEIGHT, 1)
    grid0 = torch.cat((y.unsqueeze(2), x.unsqueeze(2)), 2)
    grid0.unsqueeze_(0)

    grid = torch.from_numpy(np.tile(grid0, (NUMBER_OF_IMAGES, 1, 1, 1)))

    image_small = F.grid_sample(orig, grid)

    RGBDDs[i] = image_small

res0 = np.load('results/result_GPU_12_1090_0.npy')
res1 = np.load('results/result_GPU_12_1090_1.npy')
res2 = np.load('results/result_GPU_12_1090_2.npy')

res = np.array([res0, res1, res2])

outputs0 = torch.tensor(res0).unsqueeze(0).unsqueeze(0)
outputs1 = torch.tensor(res1).unsqueeze(0).unsqueeze(0)
outputs2 = torch.tensor(res0).unsqueeze(0).unsqueeze(0)

outputs = [outputs0, outputs1, outputs2]

RGBDD = RGBDDs[0]
data = mat73.loadmat('Exp2_data/3_images/00120_K.mat')
K = data['K']
K = K/DOWNSAMPLE_COEF
K[2,2] = 1

diff0_ios =  abs(RGBDD[0, 1, :, :] - res0)
diff1_ios =  abs(RGBDD[1, 1, :, :] - res1)
diff2_ios =  abs(RGBDD[2, 1, :, :] - res2)

print("Mean absolute error of ios")
print("Image 0: ", np.mean(np.array(diff0_ios)))
print("Image 1: ", np.mean(np.array(diff1_ios)))
print("Image 2: ", np.mean(np.array(diff2_ios)))

diff0_mvs =  abs(RGBDD[0, 0, :, :] - res0)
diff1_mvs =  abs(RGBDD[1, 0, :, :] - res1)
diff2_mvs =  abs(RGBDD[2, 0, :, :] - res2)

print("Mean absolute error of MVS")
print("Image 0: ", np.mean(np.array(diff0_mvs)))
print("Image 1: ", np.mean(np.array(diff1_mvs)))
print("Image 2: ", np.mean(np.array(diff2_mvs)))



#Evaluation metrics

#Mean squared error of reprojections into the original depth maps
for i in range(NUMBER_OF_IMAGES):

    squared_error_orig_1 = (RGBDD[i,1,:,:] - RGBDD[i,6,:,:]) ** 2
    mask1 = (squared_error_orig_1 <= 1).float()
    squared_error_orig_2 = (RGBDD[i,1,:,:] - RGBDD[i,11,:,:]) ** 2
    mask2 = (squared_error_orig_2 <= 1).float()


    mse_orig = mean_squared_error(RGBDD[i,1,:,:], RGBDD[i,6,:,:]) + mean_squared_error(RGBDD[i,1,:,:], RGBDD[i,11,:,:])
    mse_orig = mse_orig/2

    mse_orig_no_out = mean_squared_error(RGBDD[i, 1, :, :]*mask1, RGBDD[i, 6, :, :]*mask1) + mean_squared_error(RGBDD[i, 1, :, :]*mask2,RGBDD[i, 11, :, :]*mask2)
    mse_orig_no_out = mse_orig_no_out / 2

    squared_error_orig_1 = (torch.tensor(res[i,:,:]) - RGBDD[i, 6, :, :]) ** 2
    mask1 = (squared_error_orig_1 <= 1).float()
    squared_error_orig_2 = (torch.tensor(res[i,:,:]) - RGBDD[i, 11, :, :]) ** 2
    mask2 = (squared_error_orig_2 <= 1).float()

    mse_res = mean_squared_error(res[i,:,:], RGBDD[i,6,:,:]) + mean_squared_error(res[i,:,:], RGBDD[i,11,:,:])
    mse_res = mse_res/2

    mse_res_no_out = mean_squared_error(torch.tensor(res[i, :, :])*mask1, RGBDD[i, 6, :, :]*mask1) + mean_squared_error(torch.tensor(res[i, :, :])*mask2, RGBDD[i, 11, :, :]*mask2)
    mse_res_no_out = mse_res_no_out / 2

    print("Mean squared error for image %d:" % i)
    print("Original: ", mse_orig)
    print("Original (no outliers): ", mse_orig_no_out)
    print("Result: ", mse_res)
    print("Result (no outliers): ", mse_res_no_out)
    print("\n")

"""
fig, ax = plt.subplots()
im = plt.imshow(RGBDD[0,2,:,:], interpolation='nearest')
fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()
fig, ax = plt.subplots()
im = plt.imshow(RGBDD[0,7,:,:], interpolation='nearest')
fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()
"""
print("\n\n")
#Mean squared error of reprojections into the generated depth maps
x, y = np.meshgrid(np.arange(0.5, NEW_WIDTH + 0.5), np.arange(0.5, NEW_HEIGHT + 0.5))
output_reprojections_loss = 0
output_reprojections_loss_no_out = 0
cam_params = extrinsics[0]
for i in range(len(outputs)):
    for j in range(len(outputs)):
        if i != j:
            uv_from_ref = reproject(x, y, outputs[j][0, 0, :, :].float(), cam_params[j], cam_params[i], K)
            uv_from_ref = uv_from_ref[::-1]
            uv_from_ref_reshaped = uv_from_ref.reshape((2, NEW_HEIGHT, NEW_WIDTH)).transpose(1, 2, 0)
            uv_from_ref_reshaped = torch.from_numpy(uv_from_ref_reshaped.copy())
            channel_to_interpolate = outputs[i][0][0].float()
            channel_to_interpolate = channel_to_interpolate[None, None, :, :]
            ref_from_src_depth_ios = interpolate(channel_to_interpolate, uv_from_ref_reshaped)

            orig = outputs[j][0, 0, :, :]
            generated = ref_from_src_depth_ios[0, 0, :, :]

            squared_error = (orig - generated) ** 2
            mask2 = (squared_error <= 1).float()

            mask = (generated != 0).float()  # find relevant (nonzero) values
            criterion = torch.nn.MSELoss()

            loss2 = criterion(orig * mask * mask2, generated * mask * mask2)
            output_reprojections_loss_no_out += loss2

            loss = criterion(orig * mask, generated * mask)
            output_reprojections_loss += loss
            """
            fig, ax = plt.subplots()
            im = plt.imshow(orig, interpolation='nearest')
            fig.colorbar(im, ax=ax, label='Interactive colorbar')
            plt.title("Original")
            plt.show()
            fig, ax = plt.subplots()
            im = plt.imshow(generated, interpolation='nearest')
            fig.colorbar(im, ax=ax, label='Interactive colorbar')
            plt.title("Generated")
            plt.show()
            """
    print("The output reprojections loss for generated is", output_reprojections_loss/2," for image ",i)
    print("The output reprojections loss (no outlier) for generated is", output_reprojections_loss_no_out/2, " for image ", i)

output_reprojections_loss = 0
output_reprojections_loss_no_out = 0
for i in range(len(outputs)):
    for j in range(len(outputs)):
        if i != j:
            uv_from_ref = reproject(x, y, RGBDD[j, 1, :, :].float(), cam_params[j], cam_params[i], K)
            uv_from_ref = uv_from_ref[::-1]
            uv_from_ref_reshaped = uv_from_ref.reshape((2, NEW_HEIGHT, NEW_WIDTH)).transpose(1, 2, 0)
            uv_from_ref_reshaped = torch.from_numpy(uv_from_ref_reshaped.copy())
            channel_to_interpolate = RGBDD[i, 1, :, :].float()
            channel_to_interpolate = channel_to_interpolate[None, None, :, :]
            ref_from_src_depth_ios = interpolate(channel_to_interpolate, uv_from_ref_reshaped)

            orig = RGBDD[j, 1, :, :]
            generated = ref_from_src_depth_ios[0, 0, :, :]

            squared_error = (orig - generated) ** 2
            mask2 = (squared_error <= 1).float()

            mask = (generated != 0).float()  # find relevant (nonzero) values
            criterion = torch.nn.MSELoss()

            loss2 = criterion(orig * mask * mask2, generated * mask * mask2)
            output_reprojections_loss_no_out += loss2

            loss = criterion(orig * mask, generated * mask)
            output_reprojections_loss += loss
            """
            fig, ax = plt.subplots()
            im = plt.imshow(orig, interpolation='nearest')
            fig.colorbar(im, ax=ax, label='Interactive colorbar')
            plt.title("Original")
            plt.show()
            fig, ax = plt.subplots()
            im = plt.imshow(generated, interpolation='nearest')
            fig.colorbar(im, ax=ax, label='Interactive colorbar')
            plt.title("Generated")
            plt.show()
            """

    print("The output reprojections loss for original is",output_reprojections_loss/2, " for image ",i)
    print("The output reprojections loss no outliers for original is", output_reprojections_loss_no_out/2, " for image ", i)

print("\n\n")
#SSIM
for i in range(NUMBER_OF_IMAGES):
    ssim_index = ssim(np.array(res[i,:,:]), np.array(RGBDD[i,1,:,:]))
    print(f"SSIM: {ssim_index}")






"""
fig, ax = plt.subplots()
im = plt.imshow(diff, interpolation='nearest')
fig.colorbar(im, ax=ax, label='Interactive colorbar')
plt.show()
print("lol")
"""
