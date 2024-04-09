# Depth_consistency

EXPERIMENT 1
1. run generate_data.m - This creates all the data from reprojecting images (be sure to set all the variables on the first 25 lines)
2. run generate_training_rays.m - This takes the valid training rays and saves them as xxx_ready.mat (change first line with corresponding generated file from generate_data.m)
3. run generate_test_img.m - This generates the test image for visualization (choose image on line 3)

4. Load all xxx_ready.mat files into main.py and start training
5. Run test_data.py to save a .npy file that can be visualized

EXPERIMENT 2
1. run generate_data_exp2_V2.m -> this script generates xxx_RGBD.mat, xxx_cam_params.mat and xxx_K.mat (be sure to set all the variables on the first 25 lines)
2. For training, generate all required data, and run exp2_GPU_downsample_edited_with_output_reprojections.py 
3. To validate run test2_edited.py with the saved model in .pth format
4. To analyze generated .npy files run analyze_results.py
