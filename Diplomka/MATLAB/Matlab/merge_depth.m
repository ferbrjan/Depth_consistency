close all;
clear;

% load common data
rgb_image_names = dir('images/*.jpg');
images = read_images('.');

% % 1) overlay the depth image with RGB image  --> fit ok
% for i = 1:1 %size(rgb_image_names,1)
%     rgb = imread(fullfile('images',rgb_image_names(i).name));
%     load(fullfile('depth',[rgb_image_names(i).name(1:end-3) 'mat']));
%     fit_depth = imresize(depth, [size(rgb,1), size(rgb,2)]);
%     fit_depth2show = fit_depth - min(min(fit_depth));
%     fit_depth2show = double(fit_depth2show / max(max(fit_depth2show)));
%     figure(); imshow(rgb);
%     figure(); imshow(fit_depth2show);
% end 


% 2) plot the common point cloud + camera coordinate systems
gt_pts = pcread('2022-09-14-13-51-28_PointCloud.ply');
gt_pts = gt_pts.Location * diag([-1 1 1]);
rgbtxt = 'rgb';
s = 0.5;
f1 = subfig(1,1,1); xlabel('x'); ylabel('y'); zlabel('z'); hold on;   %pcshow(pointCloud(gt_pts));
for i = 1:5  %:size(rgb_image_names,1)%91 
    img = images(str2num(rgb_image_names(i).name(1:end-4)));
	Rz180 = [-1 0 0; 0 -1 0; 0 0 1];
    xR = img.R' * Rz180; % * Rx180; 
    C = diag([-1 1 1]) * img.t;  % img.t is camera center !!!
    plot3(C(1), C(2), C(3), 'r.', 'MarkerSize', 10);
    text(C(1), C(2), C(3), rgb_image_names(i).name(1:end-4), 'Color', [1 1 1]);
    for j = 1:3
    	plot3([C(1) C(1)+s*xR(1,j)], [C(2) C(2)+s*xR(2,j)], [C(3) C(3)+s*xR(3,j)], [rgbtxt(j) '-'], 'LineWidth', 2);
    end
end
    
    
% 3) the images fit well -> use the focal length from RGB image
% K = [1584.753120 0 943.874516; 0 1570.358042 700.532434; 0 0 1];
K = [1589.302 0 950.183; 0 1589.302 714.7717; 0 0 1];
rd = [0.144018 -0.278299 -0.005292 -0.000228];
N = size(rgb_image_names,1);
pts = cell(1,N);
cols = cell(1,N);
colors = getColors(N);
for i = 1:N
    img_id = i; %1 + (i-1)*10;
    img = images(str2num(rgb_image_names(img_id).name(1:end-4)));
    rgb = imread(fullfile('images',rgb_image_names(img_id).name));
    load(fullfile('depth',[rgb_image_names(img_id).name(1:end-3) 'mat']));
    fit_depth = imresize(depth, [size(rgb,1), size(rgb,2)]);
    fit_depth2show = fit_depth - min(min(fit_depth));
    fit_depth2show = double(fit_depth2show / max(max(fit_depth2show)));
    
    BW = double(edge(imresize((fit_depth2show),(1/8)*[size(rgb,1), size(rgb,2)]),"approxcanny"));
    BW2 = imresize(BW,[size(rgb,1), size(rgb,2)],'bicubic');
    BWblur = imgaussfilt(BW2,10);
    depth_filter = BWblur > 0.001;
%     figure(); imshow(depth_filter);
    
    rgb2show = double(rgb)/255;
    rgb2show(:,:,1) = rgb2show(:,:,1) + 0.5 * depth_filter;
%     f2 = subfig(3,4,i); imshow(rgb2show); hold on;
%     figure(); imshow(fit_depth2show);
    
    % show projections to image
    X = gt_pts';
    Rz180 = [-1 0 0; 0 -1 0; 0 0 1];
    R_w2c = Rz180 * img.R; 
    C_w = diag([-1 1 1]) * img.t;
    uvl = K * (R_w2c * X - repmat(R_w2c * C_w,1,size(X,2)));--
    uvl(:,uvl(3,:)<0) = [];
    uv = h2a(uvl);
    uv(:,uv(1,:)<0) = [];
    uv(:,uv(2,:)<0) = [];
    uv(:,uv(1,:)>size(rgb,2)) = [];
    uv(:,uv(2,:)>size(rgb,1)) = [];
%     plot(uv(1,:),uv(2,:),'g.'); axis auto;
    
    
    
    [ux, uy] = meshgrid(0.5:size(rgb,2),0.5:size(rgb,1));
    m = [ux(:), uy(:), ones(length(ux(:)),1)]';
%     v = (R_w2c' * inv(K)) * m;
%     K = [1584.753120 0 943.874516; 0 1570.358042 700.532434; 0 0 1];   
    v = R_w2c' * (inv(K) * m) .* repmat(fit_depth(:)',3,1);
%     figure(); pcshow(pointCloud(v')); axis equal;
    
    pts_all = C_w + v;  
    pts_all(:,depth_filter(:)) = [];
    pts{i} = pts_all(:,1:500:end);
    
    mat2vec = @(M) M(:);
    vrgb = [mat2vec(rgb(:,:,1))'; mat2vec(rgb(:,:,2))'; mat2vec(rgb(:,:,3))'];
    vrgb(:,depth_filter(:)) = [];
    cols{i} = vrgb(:,1:500:end);   %repmat(colors{i}',1,size(pts{i},2));
end 

X = cell2mat(pts);
c = cell2mat(cols);
% figure(f1); hold on; pcshow(pointCloud(X', 'Color', c'));
pcwrite(pointCloud(X', 'Color', c'), 'composed_depthmaps_01.ply')
