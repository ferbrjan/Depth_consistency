close all;
clear;

%% load common data
rgb_imags_dir = './undistorted/images/';
depth_imags_dir = './undistorted/depth/';
[cameras, images, points3D] = read_model('./undistorted/sparse');
c_images = images.values();

near_plane = 0.5;
far_plane = 5;

%% test the alignment of the depth and RGB
% t = 1;
% for i = 1:10:40
%     img = c_images{i};
%     img_name = strtrim(img.name);
%     rgb = imread(fullfile(rgb_imags_dir,img_name));
%     load(fullfile(depth_imags_dir,[img_name(1:end-3) 'mat']));
%     depth2show = double(depth / max(depth(:)));
%     rgb_show = 0.1 * double(rgb)/255;
%     rgb_show(:,:,1) = rgb_show(:,:,1) + 0.9 * depth2show;
%     rgb_show(:,:,2) = rgb_show(:,:,2) + 0.9 * depth2show;
%     rgb_show(:,:,3) = rgb_show(:,:,3) + 0.9 * depth2show;
%     subfig(3,4,t); imshow(rgb_show);
% %     subfig(3,4,t); imshow(depth2show);
% %     subfig(3,4,t); imshow(rgb);
%     t = t + 1;
% end


%% show the depthmaps in world using the COLMAP camera poses 
rgbtxt = 'rgb'; Rz180 = [-1 0 0; 0 -1 0; 0 0 1]; s = 0.5;

% depthmaps
t = 1;
step = 10;
step2 = 500;
pts = {};
cols = {};

% camera coordinates (using COLMAP definition of coordinate system and its intrinsics)
cam = cameras(1); 
[ux, uy] = meshgrid(0.5:cam.width,0.5:cam.height);
m = [ux(:), uy(:), ones(length(ux(:)),1)]'; 
K_colmap = [cam.params(1) 0 cam.params(3); 0 cam.params(2) cam.params(4); 0 0 1];
iKm = inv(K_colmap) * m;
    
c_images = images.values();
c_points3D = points3D.values();
X = cell2mat(cellfun(@(x) x.xyz, c_points3D, 'UniformOutput', false));
X = X(:,1:step:end);
for i = 1:size(c_images,2)
    img = c_images{i};
    img_name = strtrim(img.name);
    rgb = imread(fullfile(rgb_imags_dir,img_name));
    load(fullfile(depth_imags_dir,[img_name(1:end-3) 'mat']));

    fit_depth2show = depth - min(min(depth));
    fit_depth2show = double(fit_depth2show / max(max(fit_depth2show)));
    BW = double(edge(imresize((fit_depth2show),(1/8)*[size(rgb,1), size(rgb,2)]),"approxcanny",0.02));  %     figure(); imshow(BW);
    BW2 = imresize(BW,[size(rgb,1), size(rgb,2)],'bicubic');
    BWblur = imgaussfilt(BW2,20);
    depth_filter = BWblur > 0.05;
    
    filter = logical((depth(:) < near_plane) + (depth(:) > far_plane) + depth_filter(:));
    filter2show = reshape(filter,size(rgb,1),size(rgb,2));
    rgb2show = double(rgb)/255;
    rgb2show(:,:,1) = rgb2show(:,:,1) + 0.25 * filter2show;
    rgb2show(:,:,2) = rgb2show(:,:,2) + 0.25 * filter2show;
    f2 = subfig(3,4,t+2); imshow(rgb2show); hold on;

    % show projections to image (colmap)
    uv_colamp = K_colmap * (img.R * X + repmat(img.t,1,size(X,2)));
    uv_colamp = filter_uvs(uv_colamp, rgb);
    plot(uv_colamp(1,:),uv_colamp(2,:),'g.','MarkerSize',5); axis auto;
    
    % add depthmap into the cache 
    v = img.R' * (iKm .* repmat(depth(:)',3,1));
    pts_all = repmat(- img.R' * img.t,1,size(v,2)) + v;  
    pts_all(:,filter) = [];
    pts{t} = pts_all(:,1:step2:end);
    
    mat2vec = @(M) M(:);
    vrgb = [mat2vec(rgb(:,:,1))'; mat2vec(rgb(:,:,2))'; mat2vec(rgb(:,:,3))'];
    vrgb(:,filter) = [];
    cols{t} = vrgb(:,1:step2:end); 
    
    t = t + 1;
end

X = cell2mat(pts);
c = cell2mat(cols);
pcwrite(pointCloud(X', 'Color', c'), 'composed_depthmaps_COLMAP_undistorted_05-5cut.ply')

