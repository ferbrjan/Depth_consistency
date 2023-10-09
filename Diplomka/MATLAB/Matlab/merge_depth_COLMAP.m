close all;
clear;

%% load common data
rgb_image_names = dir('images/*.jpg');
images_iOS = read_images('.');
[cameras, images, points3D] = read_model('./colmap');

% K = [1584.753120 0 943.874516; 0 1570.358042 700.532434; 0 0 1]; % calib
K = [1589.302 0 950.183; 0 1589.302 714.7717; 0 0 1];   % ARFoundation

%% align models
N = size(rgb_image_names,1);
C_iOS = [];
C_colamp = [];
iOS_name2id = images_name2id(images_iOS);
colmap_name2id = images_name2id(images);
for i = 1:N  
    image_name = strtrim(rgb_image_names(i).name);
    if isKey(iOS_name2id,['images/' image_name]) && isKey(colmap_name2id,image_name)
        img_iOS = images_iOS(iOS_name2id(['images/' image_name]));
        C_iOS(:,end+1) = diag([-1 1 1]) * img_iOS.t;

        img_colmap = images(colmap_name2id(image_name));
        C_colamp(:,end+1) = - img_colmap.R' * img_colmap.t;
    end
end
% subfig(3,4,1); 
% plot3(C_iOS(1,:), C_iOS(2,:), C_iOS(3,:), 'r.', 'MarkerSize', 7); hold on;
% plot3(C_colamp(1,:), C_colamp(2,:), C_colamp(3,:), 'b.', 'MarkerSize', 7);

%   [D, Z, TRANSFORM] = PROCRUSTES(X, Y)
[D, Z, TRANSFORM] = procrustes(C_iOS', C_colamp', 'Reflection', false);
s_c2i = TRANSFORM.b; 
R_c2i = TRANSFORM.T';
t_c2i = TRANSFORM.c(1,:)';
C_colmap_in_iOS = s_c2i * R_c2i * C_colamp + repmat(t_c2i,1,size(C_colamp,2));
% plot3(C_colmap_in_iOS(1,:), C_colmap_in_iOS(2,:), C_colmap_in_iOS(3,:), 'yo', 'MarkerSize', 7);

pts_dist_in_cm = 100*sqrt(sum((C_colmap_in_iOS - C_iOS).^2));
fprintf('Mean distance is %.2f [cm] with std %.2f [cm]\n', mean(pts_dist_in_cm), std(pts_dist_in_cm));

subfig(3,4,1); histogram(pts_dist_in_cm, 100);
title('Distances between COLMAP and the iOS camera centers'); xlabel('Distance in [centimeters]'); ylabel('Occurence');

%% transform the whole colmap model to iOS, i.e., convert the camera poses
c_images = images.values();
for i = 1:size(c_images,2) 
    img = c_images{i};
    C_c2i = s_c2i * R_c2i * (- img.R' * img.t) + t_c2i;
    img.R = img.R * R_c2i';
    img.t = - img.R * C_c2i;
    img_name = strtrim(img.name);            
    img.name = ['images/' img_name];     
    images(img.image_id) = img;
end
c_points3D = points3D.values();
for i = 1:size(c_points3D,2) 
    pt = c_points3D{i};
    pt.xyz = s_c2i * R_c2i * pt.xyz + t_c2i;
    points3D(pt.point3D_id) = pt;
end
if ~exist('colmap_in_iOS','dir')
    mkdir('colmap_in_iOS')
end
saveColmap( './colmap_in_iOS', cameras, images, points3D ); 


%% show the depthmaps in world using the COLMAP camera poses 
X_iOS = pcread('2022-09-14-13-51-28_PointCloud.ply');
X_iOS = diag([-1 1 1]) * X_iOS.Location';
rgbtxt = 'rgb'; Rz180 = [-1 0 0; 0 -1 0; 0 0 1]; s = 0.5;

c_images = images.values();
c_points3D = points3D.values();
C = cell2mat(cellfun(@(im) - im.R' * im.t, c_images, 'UniformOutput', false));
X = cell2mat(cellfun(@(x) x.xyz, c_points3D, 'UniformOutput', false));

f1 = subfig(3,4,2); pcshow(pointCloud(X_iOS', 'Color', repmat([1 0 0],size(X_iOS,2),1))); hold on;
plot3(C_iOS(1,:), C_iOS(2,:), C_iOS(3,:), 'rx', 'MarkerSize', 7, 'LineWidth', 2); hold on;
plot3(C_colmap_in_iOS(1,:), C_colmap_in_iOS(2,:), C_colmap_in_iOS(3,:), 'go', 'MarkerSize', 7, 'LineWidth', 2); hold on;
pcshow(pointCloud(X', 'Color', repmat([0 1 0],size(X,2),1))); 
xlabel('x'); ylabel('y'); zlabel('z'); axis([-5 15 -5 3 -10 10]); hold on;

% coordintes in world
for i = 1:5:size(c_images,2)
    img = c_images{i};
    img_name = strtrim(img.name);
    img_iOS = images_iOS(iOS_name2id(['images/' img_name]));
    R_iOS = img_iOS.R' * Rz180;  
    C_iOS = diag([-1 1 1]) * img_iOS.t;  % img.t is camera center !!!
    text(C_iOS(1), C_iOS(2), C_iOS(3), img_name, 'Color', [1 1 1]); hold on;
    for j = 1:3
    	plot3([C_iOS(1) C_iOS(1)+s*R_iOS(1,j)], ...
            [C_iOS(2) C_iOS(2)+s*R_iOS(2,j)], ...
            [C_iOS(3) C_iOS(3)+s*R_iOS(3,j)], [rgbtxt(j) '-'], 'LineWidth', 2);  
        hold on;
    end
    
    R = img.R;
    C = - R' * img.t;
%     text(C(1), C(2), C(3), img_name, 'Color', [1 1 1]); hold on;
    for j = 1:3
    	plot3([C(1) C(1)+s*R(1,j)], ...
            [C(2) C(2)+s*R(2,j)], ...
            [C(3) C(3)+s*R(3,j)], [rgbtxt(j) '-'], 'LineWidth', 2);  
        hold on;
    end
end


% depthmaps
t = 1;
step = 10;
step2 = 500;
pts = {};
cols = {};
X = X(:,1:step:end);
X_iOS = X_iOS(:,1:step:end);

% camera coordinates (using COLMAP definition of coordinate system and its intrinsics)
cam = cameras(1); 
[ux, uy] = meshgrid(0.5:cam.width,0.5:cam.height);
m = [ux(:), uy(:), ones(length(ux(:)),1)]'; 
K_colmap = [cam.params(1) 0 cam.params(2); 0 cam.params(1) cam.params(3); 0 0 1];
iKm = inv(K_colmap) * m;
    
for i = 1:size(c_images,2)
    img = c_images{i};
    img_name = strtrim(img.name);
    rgb = imread(fullfile('images',img_name));
    load(fullfile('depth',[img_name(1:end-3) 'mat']));
    fit_depth = imresize(depth, [size(rgb,1), size(rgb,2)]);
%     imwrite(uint16(1000*fit_depth),['./depth/' img_name(1:end-3) 'jpg'], 'BitDepth', 16);
    
    fit_depth2show = fit_depth - min(min(fit_depth));
    fit_depth2show = double(fit_depth2show / max(max(fit_depth2show)));
    BW = double(edge(imresize((fit_depth2show),(1/8)*[size(rgb,1), size(rgb,2)]),"approxcanny"));
    BW2 = imresize(BW,[size(rgb,1), size(rgb,2)],'bicubic');
    BWblur = imgaussfilt(BW2,10);
    depth_filter = BWblur > 0.001;
%     figure(); imshow(fit_depth2show);

%     rgb2show = double(rgb)/255;
%     rgb2show(:,:,1) = rgb2show(:,:,1) + 0.25 * depth_filter;
%     rgb2show(:,:,2) = rgb2show(:,:,2) + 0.25 * depth_filter;
%     f2 = subfig(3,4,t+2); imshow(rgb2show); hold on;
%     
%     % show projections to image (iOS)
%     img_iOS = images_iOS(iOS_name2id(['images/' img_name]));
%     R_iOS = Rz180 * img_iOS.R;  
%     C_iOS = diag([-1 1 1]) * img_iOS.t; 
%     uv_iOS = K * (R_iOS * X_iOS - repmat(R_iOS * C_iOS,1,size(X_iOS,2)));
%     uv_iOS = filter_uvs(uv_iOS, rgb);
%     plot(uv_iOS(1,:),uv_iOS(2,:),'r.','MarkerSize',10); hold on;
%     
%     % show projections to image (iOS using colmap poses)
%     uv_c2i =  K * (img.R * X_iOS + repmat(img.t,1,size(X_iOS,2)));
%     uv_c2i = filter_uvs(uv_c2i, rgb);
%     plot(uv_c2i(1,:),uv_c2i(2,:),'b.','MarkerSize',10); hold on;
%     
%     % show projections to image (colmap)
%     uv_colamp = K * (img.R * X + repmat(img.t,1,size(X,2)));
%     uv_colamp = filter_uvs(uv_colamp, rgb);
%     plot(uv_colamp(1,:),uv_colamp(2,:),'g.','MarkerSize',10); axis auto;
    
    % add depthmap into the cache
    v = img.R' * (iKm .* repmat(fit_depth(:)',3,1));
    pts_all = repmat(- img.R' * img.t,1,size(v,2)) + v;  
    pts_all(:,depth_filter(:)) = [];
    pts{t} = pts_all(:,1:step2:end);
    
    mat2vec = @(M) M(:);
    vrgb = [mat2vec(rgb(:,:,1))'; mat2vec(rgb(:,:,2))'; mat2vec(rgb(:,:,3))'];
    vrgb(:,depth_filter(:)) = [];
    cols{t} = vrgb(:,1:step2:end); 
    
    t = t + 1;
end

% X = cell2mat(pts);
% c = cell2mat(cols);
% pcwrite(pointCloud(X', 'Color', c'), 'composed_depthmaps_COLMAP.ply')

