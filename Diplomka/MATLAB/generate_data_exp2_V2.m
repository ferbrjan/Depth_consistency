clear
close all
addpath('./Matlab')
root_dir = './Example_2022-09-14-13-51-28';

if strcmp(root_dir,"./test2")
    test2 = true; 
else
    test2 = false;
end

if test2
    trim = 21;
else
    trim = 8;
end

% setting 
n = 2;                           %number of images
k = 100;                         %number of intersection points
ref_image_name = "00012.jpg";    %reference image name
no_outlier_thresh = 0.05;        %threshold for classifiing sample as not containing outliers if < no_outlier_thresh
outlier_thresh = 0.2;            %threshold for classifiing sample as containg outliers if >outlier_thresh


%% read COLMAP model
[cameras, images, points3D] = read_model(root_dir);
image_vals = images.values();
camera_vals = cameras.values();
points_vals = points3D.values(); 
im_w = camera_vals{1}.width;
im_h = camera_vals{1}.height;

% compose calib. matrix - assume one camera model
K = eye(3);
K(1,1) = camera_vals{1}.params(1);
K(2,2) = camera_vals{1}.params(2);
K(1,3) = camera_vals{1}.params(3);      %-10
K(2,3) = camera_vals{1}.params(4);      %-10

% find number of common 3D points for image pairs
L = size(image_vals,2);
G = zeros(L,L);
for i = 1:L
    for j = i+1:L
       G(i,j) = sum(intersect(image_vals{i}.point3D_ids, image_vals{j}.point3D_ids) ~= -1); 
    end
end

fprintf("\n\n\nReading complete!\n\n\n")


%% load images
img_val_id = -1;
for i = 1:L  % find reference image
    if strcmp(ref_image_name, strtrim(image_vals{i}.name(trim:end)))
        img_val_id = i;
        ref_img = image_vals{i};
        break;
    end
end
if test2
    [ref_rgb, ref_depth_ios, ref_depth_mvs] = copy_load_images(root_dir, ref_image_name); %load_images
    ref_depth_ios = imresize(ref_depth_ios,3.75);
    ref_depth_mvs = imresize(ref_depth_mvs,3.75);
else
    [ref_rgb, ref_depth_ios, ref_depth_mvs] = load_images(root_dir, ref_image_name);
end
src = struct();
[vals, ids] = sort(G(img_val_id,:),'descend');
if vals(n) < k
    error('Too small images overlap!');
end
for i = 1:n         
    src(i).img = image_vals{ids(i)};
    if test2
        [src(i).rgb, src(i).depth_ios, src(i).depth_mvs] = copy_load_images(root_dir, strtrim(src(i).img.name(trim:end))); 
        src(i).depth_ios = imresize(src(i).depth_ios,3.75);
        src(i).depth_mvs = imresize(src(i).depth_mvs,3.75);
    else
        [src(i).rgb, src(i).depth_ios, src(i).depth_mvs] = load_images(root_dir, strtrim(src(i).img.name(trim:end))); 
    end
end

%% Create data pool for all n+1 images
pool = struct();
for i = 1:n
    pool(i).img=src(i).img;
    pool(i).rgb=src(i).rgb;
    pool(i).depth_ios = src(i).depth_ios;
    pool(i).depth_mvs = src(i).depth_mvs;
end
pool(end+1).rgb = ref_rgb;
pool(end).depth_ios = ref_depth_ios;
pool(end).depth_mvs = ref_depth_mvs;
pool(end).img = ref_img;

%% Make one image reference and the other source using the created pool
ref_src = (1:n+1);
output_full = cell(n+1,1);
output_params =  cell(n+1,1);

for ref_image = 1:n+1

    ref_src = circshift(ref_src,1);
    
    ref_img = pool(ref_src(1)).img;
    ref_depth_mvs = pool(ref_src(1)).depth_mvs;
    ref_depth_ios = pool(ref_src(1)).depth_ios;
    ref_rgb = pool(ref_src(1)).rgb;

    for i =2:n+1
        src(i-1).img=pool(ref_src(i)).img;
        src(i-1).rgb=pool(ref_src(i)).rgb;
        src(i-1).depth_ios=pool(ref_src(i)).depth_ios;
        src(i-1).depth_mvs=pool(ref_src(i)).depth_mvs;
    end


    %% find transform. params 
    ref_E = eye(4);
    ref_E(1:3,1:3) = ref_img.R;
    ref_E(1:3,4)= ref_img.t;
    output_params{ref_image,1} =  ref_E;
    for i = 1:n         
        src_E = eye(4);
        src_E(1:3,1:3) = src(i).img.R;
        src_E(1:3,4) = src(i).img.t;
        src(i).E = src_E;
    end
    
    ref_C = -ref_E(1:3,1:3)'*ref_E(1:3,4);

    %% Reproject to ref
    reproject = @(x,y,d,E_ref,E_src) h2a(K * h2a(E_src * inv(E_ref) *a2h(inv(K) * a2h([x(:),y(:)]') .* repmat(d(:)',3,1))));
    
    output = zeros(im_h,im_w,5*(n+1));

    [x,y]=meshgrid(0.5:im_w,0.5:im_h);

    output(:,:,1) = ref_depth_mvs;
    output(:,:,2) = ref_depth_ios;
    output(:,:,3:5) = cast(ref_rgb,"single");

    for i = 1:n
        uv_from_ref = reproject(x,y,ref_depth_ios,ref_E,src(i).E);

        ref_from_src_depth_ios = interp2(x, y, src(i).depth_ios, ...
        reshape(uv_from_ref(1,:),im_h,im_w), reshape(uv_from_ref(2,:),im_h,im_w), "cubic"); 
        ref_from_src_depth_mvs = interp2(x, y, src(i).depth_mvs, ...
        reshape(uv_from_ref(1,:),im_h,im_w), reshape(uv_from_ref(2,:),im_h,im_w), "cubic");
        ref_from_src_rgb = zeros(im_h,im_w,3);
        for j = 1:3
            ref_from_src_rgb(:,:,j) = interp2(x, y, double(src(i).rgb(:,:,j)), ...
            reshape(uv_from_ref(1,:),im_h,im_w), reshape(uv_from_ref(2,:),im_h,im_w), "cubic");
        end
        output(:,:,1 + (i*5)) = ref_from_src_depth_mvs;
        output(:,:,2 + (i*5)) = ref_from_src_depth_ios;
        output(:,:,(3 + (i*5)):(5 + (i*5))) = round(cast(ref_from_src_rgb,"single"));
        
        %% Check visualization - looks gucci

        %RGB 
        %{
        figure()
        imshow(ref_rgb)
        figure()
        imshow(src(i).rgb)
        figure()
        imshow(uint8(ref_from_src_rgb))
        figure()
        imshow(uint8(0.5*ref_rgb + uint8(0.5*ref_from_src_rgb)))
        close all; %Add breakpoint here
        
        %DEPTH
        ref_from_src_rgb(isnan(ref_from_src_rgb))=0;
        figure()
        imagesc(ref_depth_ios)
        figure()
        imagesc(src(i).depth_ios)
        figure()
        imagesc(ref_from_src_depth_ios)
        figure()
        imagesc(0.5* double(ref_depth_ios) + 0.5*double(ref_from_src_depth_ios));
        close all; %Add breakpoint here
        %}
        
        
    end
    output_full{ref_image} = output;
end

save(erase(join([erase(ref_image_name,".jpg"),"_RGBD.mat"])," "),"output_full",'-v7.3')
save(erase(join([erase(ref_image_name,".jpg"),"_cam_params.mat"])," "),"output_params",'-v7.3')
save(erase(join([erase(ref_image_name,".jpg"),"_K.mat"])," "),"K",'-v7.3')
