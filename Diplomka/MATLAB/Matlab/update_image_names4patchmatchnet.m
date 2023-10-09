close all;
clear;

rel_img_folder = 'images';

% [cameras, images, points3D] = read_model('./undistorted/sparse');
[cameras, images, points3D] = read_model('./colmap_in_iOS');
c_images = images.values();
for i = 1:size(c_images,2)
    img = c_images{i};
    img_name = strtrim(img.name);
    img.name = [rel_img_folder '/' img_name];
    images(img.image_id) = img;
end

% if ~exist('colmap_in_iOS4patchmatchnet','dir')
%     mkdir('colmap_in_iOS4patchmatchnet')
% end
saveColmap( './colmap_in_iOS', cameras, images, points3D ); 

% TODO: in COLMAP convert the TXT files into BIN files