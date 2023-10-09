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
n = 8;                         %number of images
k = 100;                        %number of intersection points
ref_image_name = "00020.jpg";   %reference image name
no_outlier_thresh = 0.05;        %threshold for classifiing sample as not containing outliers if < no_outlier_thresh
outlier_thresh = 0.2;             %threshold for classifiing sample as containg outliers if >outlier_thresh

%00075

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

%% find transform. params 
ref_E = eye(4);
ref_E(1:3,1:3) = ref_img.R;
ref_E(1:3,4)= ref_img.t;
for i = 1:n         % find reference image
    src_E = eye(4);
    src_E(1:3,1:3) = src(i).img.R;
    src_E(1:3,4) = src(i).img.t;
    src(i).E = src_E;
end

ref_C = -ref_E(1:3,1:3)'*ref_E(1:3,4);

[x,y]=meshgrid(0.5:im_w,0.5:im_h);
reproject = @(x,y,d,E_ref,E_src) h2a(K * h2a(E_src * inv(E_ref) *a2h(inv(K) * a2h([x(:),y(:)]') .* repmat(d(:)',3,1))));

%% Check 3D projections
ref_X_ios = h2a(inv(ref_E) *  a2h(inv(K) * [x(:),y(:),ones(length(x(:)),1)].' .* repmat(ref_depth_ios(:)',3,1))); 
ref_X_ios_color = [reshape(ref_rgb(:,:,1),1,[]);reshape(ref_rgb(:,:,2),1,[]);reshape(ref_rgb(:,:,3),1,[])];
ref_X_mvs = h2a(inv(ref_E) *  a2h(inv(K) * [x(:),y(:),ones(length(x(:)),1)].' .* repmat(ref_depth_mvs(:)',3,1)));
ref_X_mvs_color = [reshape(ref_rgb(:,:,1),1,[]);reshape(ref_rgb(:,:,2),1,[]);reshape(ref_rgb(:,:,3),1,[])];

PC_ios=ref_X_ios;
PC_ios_color = ref_X_ios_color;

PC_mvs = ref_X_mvs;
PC_mvs_color = ref_X_mvs_color;

%{
for i = 1:n
    src_X_ios = h2a(inv(src(i).E) *  a2h(inv(K) * [x(:),y(:),ones(length(x(:)),1)].' .* repmat(src(i).depth_ios(:)',3,1)));
    src_X_ios_color = [reshape(src(i).rgb(:,:,1),1,[]);reshape(src(i).rgb(:,:,2),1,[]);reshape(src(i).rgb(:,:,3),1,[])];
    src_X_mvs = h2a(inv(src(i).E) *  a2h(inv(K) * [x(:),y(:),ones(length(x(:)),1)].' .* repmat(src(i).depth_mvs(:)',3,1)));
    src_X_mvs_color = [reshape(src(i).rgb(:,:,1),1,[]);reshape(src(i).rgb(:,:,2),1,[]);reshape(src(i).rgb(:,:,3),1,[])];

    PC_ios=[PC_ios,src_X_ios];
    PC_ios_color=[PC_ios_color,src_X_ios_color];
    PC_mvs=[PC_mvs,src_X_ios];
    PC_mvs_color=[PC_mvs_color,src_X_ios_color];
end
%}

PC = [PC_ios,PC_mvs];
PC_color = [PC_ios_color,PC_mvs_color];

%ptcloud = pointCloud(PC_ios',"Color",PC_ios_color');
%pcshow(ptcloud)

%% Project to 3D

%ADD HERE ALL POINTS x = [1-1440],y=[1-1920]

close all
%figure("Name","REF img");
%imagesc(ref_rgb); 
%[x,y] = ginput(1);
%x=round(x);
%y=round(y);

points_3d_mvs_cell =cell(im_h,im_w);
points_3d_ios_cell =cell(im_h,im_w);
ray_directions =  cell(im_h,im_w);
GTs = cell(im_h,im_w);
stdevs = [];
medians = [];
avgs = [];
categories = [];
for x=1:im_w %comment
    x
    for y=1:im_h %comment
        points_3d_mvs =[];
        points_3d_ios =[];
        ref_mvs_3d = h2a(inv(ref_E)*a2h(inv(K)*[x,y,1]'*ref_depth_mvs(y,x)));
        ref_ios_3d = h2a(inv(ref_E)*a2h(inv(K)*[x,y,1]'*ref_depth_ios(y,x)));
        ref_3d =  (ref_mvs_3d + ref_ios_3d)/2;
        points_3d_mvs = [points_3d_mvs,ref_mvs_3d];
        points_3d_ios = [points_3d_ios,ref_ios_3d];
        
        for i=1:n
            uv = reproject(x,y,ref_depth_ios(y,x),ref_E,src(i).E);
            if round(uv(1))>0 && round(uv(2))>0 && round(uv(2))<im_h && round(uv(1))<im_w
                depth_mvs_src = src(i).depth_mvs(round(uv(2)),round(uv(1)));
                depth_ios_src =  src(i).depth_ios(round(uv(2)),round(uv(1)));
                mvs_3d =  h2a(inv(src(i).E)* a2h(inv(K)*[uv(1),uv(2),1]'*depth_mvs_src));
                ios_3d =  h2a(inv(src(i).E)* a2h(inv(K)*[uv(1),uv(2),1]'*depth_ios_src));
                points_3d_mvs = [points_3d_mvs,mvs_3d];
                points_3d_ios = [points_3d_ios,ios_3d];
            end
        
        end
        %comment
        points_3d_mvs_cell{y,x}=points_3d_mvs; 
        points_3d_ios_cell{y,x}=points_3d_ios;
        ray_directions{y,x} = ref_3d -ref_C;
        [points_on_ray,distances_on_ray] = project_2_line(ray_directions{y,x}',ref_C',[points_3d_ios,points_3d_mvs]');
        points_on_ray = points_on_ray';
        GTs{y,x} = distances_on_ray';
        stdevs(y,x) = std(distances_on_ray);
        avgs(y,x) = mean(distances_on_ray);
        medians(y,x) = median(distances_on_ray);
        categories(y,x) = 0;                                                   %1 - no outliers 
        if stdevs(y,x) < no_outlier_thresh
         categories(y,x) = 1;
        elseif stdevs(y,x) >= no_outlier_thresh && stdevs(y,x)<outlier_thresh       %2 - unknown
          categories(y,x) =2;
        else                                                            %3 - outliers 
          categories(y,x) =3;
        end
        %comment
    end%comment
end%comment
%{
ray_direction = ref_3d-ref_C;
[points_on_ray,distances_on_ray] = project_2_line(ray_direction',ref_C',[points_3d_ios,points_3d_mvs]');
points_on_ray = points_on_ray';
GT = distances_on_ray';

edges = 0:0.1:20;
histogram = histcounts(distances_on_ray,edges);

stdev = std(distances_on_ray);
avg = mean(distances_on_ray);

category = 0;                                                   %1 - no outliers 
if stdev < no_outlier_thresh
 category = 1;
 color = "g";
 fprintf("\n\nPoint classified as NO OUTLIERS\n\n")
elseif stdev >= no_outlier_thresh && stdev<outlier_thresh       %2 - unknown
  category =2;
  color = "y";
  fprintf("\n\nPoint classified as UNKNWON\n\n")
else                                                            %3 - outliers 
  category =3;
  color = "r";
  fprintf("\n\nPoint classified as CONTAINING OUTLIERS\n\n")
end
%}

save(erase(join([erase(ref_image_name,".jpg"),".mat"])," "),"ray_directions","GTs","stdevs","avgs","categories","medians")
%% Generate sample with outliers
num_samples = 20;
distances = abs(2*stdev * randn(1, num_samples) + avg);
disp("Mean disatnce in GT sample is:")
avg
disp("Standard deviation of GT sample is:")
stdev
out = [GT,distances];

 %% Convert to 0/1 representation
GT_bin = convertDistancesToBinary(GT,0.01,10);
out_bin = convertDistancesToBinary(out,0.01,10);
disp("Distances contained in GT vector")
fprintf('%s\n\n',sprintf('  %.2f',find(GT_bin==1)*0.01))
disp("Distances contained in OUT vector")
fprintf('%s\n\n',sprintf('  %.2f',find(out_bin==1)*0.01))

%% Visualize
figure("Name","Point projections visualized")
ptcloud = pointCloud([PC_ios]',"Color",[PC_ios_color]');
pcshow(ptcloud);
hold on;

plot3(points_3d_ios(1,:),points_3d_ios(2,:),points_3d_ios(3,:),"r.","MarkerSize",10);
plot3(points_3d_mvs(1,:),points_3d_mvs(2,:),points_3d_mvs(3,:),"y.","MarkerSize",10);

plot3(ref_C(1),ref_C(2),ref_C(3),"g.",MarkerSize=30);
plot3([ref_C(1),ref_C(1)+5*(ray_direction(1))] , [ref_C(2),ref_C(2)+5*(ray_direction(2))],[ref_C(3),ref_C(3)+5*(ray_direction(3))],"g");

plot3(points_on_ray(1, :), points_on_ray(2, :), points_on_ray(3, :), 'wx', "MarkerSize",10);

legend("","Points ios","Points mvs","Camera center","Ray","Points projected to ray","TextColor","white")

figure("Name","Distances on ray")
hAxes = axes('NextPlot','add',...          
             'DataAspectRatio',[1 1 1],...  
             'XLim',[0,10],...               
             'YLim',[0 eps],...             
             'Color','none');  

 plot([distances_on_ray;distances'],0,"Marker","*","color",color,'MarkerSize',10);  

%% NOTES

% JAK VYTVOŘIT OUTLIER SAMPLE

% 1) - Vemu paprsek z kategorie 1 jako GT -> (spočítám si mean a std a uložim si
% to jako GT)

% 2) Vemu paprsek z kategorie 3 jako outlier
% 3) Spočítám si median / mean a zarovnám tak aby ty průměry seděli na sebe
% 4) Joinnu ty dva vzorky dohromady a subsampluju aby ten počet vzorků byl odpovídající tomu GT

% 5) Uložim si vzorky jako 0/#points pole projetý softmaxem


% JAK VYTVOŘIT VÍC OUTLIER SAMPLŮ

% 1) spočítám si ty paprsky pro každej pixel v referenci (144-171)
% 2) Nakombim dobrý a špatný a uložim si ten dobrej! + ten nakombenej
% 3) Zkusim pustit pro cca 20 random snímků 


% TEST + VALIDATE + TRAIN! sets!!!

% hubbers estimator

% zkusit softmax po centimetrech




