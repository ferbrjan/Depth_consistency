function [rgb, depth_ios, depth_mvs] = copy_load_images(root_dir, image_name)
    rgb = imread(fullfile(root_dir, 'images1', image_name));
    load(fullfile(root_dir, 'depth1', strrep(image_name,'jpg','mat')));
    depth_ios = depth;
    load(fullfile(root_dir, 'depth2', strrep(sprintf('%s%s', '', image_name),'jpg','mat')));
    depth_mvs = depth;
end

