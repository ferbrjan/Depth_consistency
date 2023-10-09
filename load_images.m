function [rgb, depth_ios, depth_mvs] = load_images(root_dir, image_name)
    rgb = imread(fullfile(root_dir, 'images', image_name));
    load(fullfile(root_dir, 'depth_ios', strrep(image_name,'jpg','mat')));
    depth_ios = depth;
    load(fullfile(root_dir, 'depth_mvs', strrep(sprintf('%s%s', '000', image_name),'jpg','mat')));
    depth_mvs = squeeze(depth_mvs);
end

