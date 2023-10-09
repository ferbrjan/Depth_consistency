close all;
clear;

%% load common data
dataset_path = '../Example_2022-09-14-13-51-28';
depth_mvs_dir = fullfile(dataset_path,'depth_mvs');
depth_ios_dir =  fullfile(dataset_path,'depth_ios');
depth_stat_dir = fullfile(dataset_path,'depth_stat');
[cameras, images, points3D] = read_model(dataset_path);
c_images = images.values();

% %% test the alignment of the depth and RGB
t = 1;
% for i = 1:10:40
%     img = c_images{i};
%     img_name = strtrim(img.name);
%     rgb = imread(fullfile(dataset_path,img_name));
%     load(fullfile(depth_ios_dir,[img_name(8:end-3) 'mat']));
%     depth_ios = depth;
%     load(fullfile(depth_mvs_dir,['000' img_name(8:end-3) 'mat']));
%     depth_mvs = reshape(depth_mvs,[size(depth_mvs,2),size(depth_mvs,3)]);
%     
%     % depth ios
%     depth2show = double(depth_ios / max(depth_ios(:)));
%     rgb_show = 0.1 * double(rgb)/255;
%     rgb_show(:,:,1) = rgb_show(:,:,1) + 0.9 * depth2show;
%     rgb_show(:,:,2) = rgb_show(:,:,2) + 0.9 * depth2show;
%     rgb_show(:,:,3) = rgb_show(:,:,3) + 0.9 * depth2show;
%     subfig(3,4,t); imshow(rgb_show);
%     t = t + 1;
%     
%     % depth mvs
%     depth2show = double(depth_mvs / max(depth_mvs(:)));
%     rgb_show = 0.1 * double(rgb)/255;
%     rgb_show(:,:,1) = rgb_show(:,:,1) + 0.9 * depth2show;
%     rgb_show(:,:,2) = rgb_show(:,:,2) + 0.9 * depth2show;
%     rgb_show(:,:,3) = rgb_show(:,:,3) + 0.9 * depth2show;
%     subfig(3,4,t); imshow(rgb_show);
%     t = t + 1;
% 
% end


%% find distances from camera to individual pixels + calculate std and mean

cam = cameras(1); 
[ux, uy] = meshgrid(0.5:cam.width,0.5:cam.height);
m = [ux(:), uy(:), ones(length(ux(:)),1)]'; 
K_colmap = [cam.params(1) 0 cam.params(3); 0 cam.params(2) cam.params(4); 0 0 1];
iKm = inv(K_colmap) * m;

for i = 1:size(c_images,2)
    fprintf('processing %d / %d\n', i, size(c_images,2))
    img = c_images{i};
    img_name = strtrim(img.name);
    rgb = imread(fullfile(dataset_path,img_name));
    load(fullfile(depth_ios_dir,[img_name(8:end-3) 'mat']));
    depth_ios = depth;
    load(fullfile(depth_mvs_dir,['000' img_name(8:end-3) 'mat']));
    depth_mvs = reshape(depth_mvs,[size(depth_mvs,2),size(depth_mvs,3)]);
    
    cam_pts_ios = (iKm .* repmat(depth_ios(:)',3,1));
    d_ios = sqrt(sum(cam_pts_ios.^2));
    cam_pts_mvs = (iKm .* repmat(depth_mvs(:)',3,1));
    d_mvs = sqrt(sum(cam_pts_mvs.^2));
    
    c2X_ios = reshape(d_ios,size(depth_ios,1),size(depth_ios,2));
    c2X_mvs = reshape(d_mvs,size(depth_mvs,1),size(depth_mvs,2));
    
%     % depth ios
%     depth2show = double(c2X_ios / max(c2X_ios(:)));
%     rgb_show = 0.1 * double(rgb)/255;
%     rgb_show(:,:,1) = rgb_show(:,:,1) + 0.9 * depth2show;
%     rgb_show(:,:,2) = rgb_show(:,:,2) + 0.9 * depth2show;
%     rgb_show(:,:,3) = rgb_show(:,:,3) + 0.9 * depth2show;
%     subfig(3,4,t); imshow(rgb_show);
%     t = t + 1;
%     
%     % depth mvs
%     depth2show = double(depth_mvs / max(depth_mvs(:)));
%     rgb_show = 0.1 * double(rgb)/255;
%     rgb_show(:,:,1) = rgb_show(:,:,1) + 0.9 * depth2show;
%     rgb_show(:,:,2) = rgb_show(:,:,2) + 0.9 * depth2show;
%     rgb_show(:,:,3) = rgb_show(:,:,3) + 0.9 * depth2show;
%     subfig(3,4,t); imshow(rgb_show);
%     t = t + 1;
    
    mean_c2X = (c2X_ios + c2X_mvs)/2;
    var_c2X = ((c2X_ios-mean_c2X).^2 + (c2X_mvs-mean_c2X).^2) / 2;
    
%     % show the variance by red coloring 
%     depth2show = double(mean_c2X / max(mean_c2X(:)));
%     vardepth2show = double(var_c2X / max(var_c2X(:)));
%     rgb_show = 0.1 * double(rgb)/255;
%     rgb_show(:,:,1) = rgb_show(:,:,1) + 0.9 * depth2show + 0.9 * vardepth2show;
%     rgb_show(:,:,2) = rgb_show(:,:,2) + 0.9 * depth2show;
%     rgb_show(:,:,3) = rgb_show(:,:,3) + 0.9 * depth2show;
%     subfig(3,4,t); imshow(rgb_show);
%     t = t + 1;
    
    save(fullfile(depth_stat_dir,[img_name(8:end-3) 'mat']), 'mean_c2X', 'var_c2X');
end










